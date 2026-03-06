"""Integrate Transformer router into paramAgent.py"""
import re
import sys

def integrate_router():
    with open('paramAgent.py', 'r') as f:
        content = f.read()
    
    # Check if already using transformer router
    if 'infer_transformer_router' in content:
        print("Already using transformer router")
        return False
    
    # 1. Update imports
    import_pattern = r'(from memory_router\.infer_router import infer as infer_router)'
    import_replacement = r'''try:
    from memory_router.infer_transformer_router import infer_transformer_router
    ROUTER_INFER_AVAILABLE = True
except ImportError:
    infer_transformer_router = None
    ROUTER_INFER_AVAILABLE = False'''
    
    content = re.sub(import_pattern, import_replacement, content)
    
    # 2. Find and update router usage in phase2 logic
    # Look for the section where router is called (around line 1000-1100)
    # The ensemble approach should be replaced with direct transformer router call
    
    # Find the select_stage2_traj_ensemble function and replace it
    ensemble_pattern = r'def select_stage2_traj_ensemble\([^)]+\)[^:]+:.*?(?=\n    def |\n\ndef |\Z)'
    
    ensemble_replacement = '''def select_stage2_traj_transformer_router(
        retrieval_pool: List[dict],
        curr_prompt_emb: Any,
        reflection_text: str,
        feedback_history: List[str],
        router_ckpt_path: str,
        router_conf_threshold: float = 0.6,
    ) -> tuple[Optional[dict], List[float], float]:
        """
        Use Transformer router to select best memory and compute fusion weights.
        
        Returns:
            (selected_traj, router_mix, router_conf)
        """
        if len(retrieval_pool) == 0:
            return None, [0.0, 0.0, 0.0], 0.0
        
        prompt_query = _to_1d_float_array(curr_prompt_emb)
        if prompt_query is None:
            return None, [0.0, 0.0, 0.0], 0.0
        
        # Compute similarity scores for all candidates
        prompt_sims = []
        reflection_sims = []
        negative_penalties = []
        
        for traj in retrieval_pool:
            # Prompt similarity
            p_sim = _safe_dot_sim(traj.get("prompt_embedding"), prompt_query)
            prompt_sims.append(p_sim)
            
            # Reflection similarity (if available)
            r_emb = traj.get("reflection_embedding")
            if r_emb is not None:
                r_sim = _safe_dot_sim(r_emb, prompt_query)
                reflection_sims.append(r_sim)
            else:
                reflection_sims.append(0.0)
            
            # Negative penalty (if trajectory is from negative bank)
            is_negative = traj.get("is_negative", False)
            negative_penalties.append(1.0 if is_negative else 0.0)
        
        # Build state dict for router
        state = {
            "attempt_count": len(feedback_history),
            "has_feedback": len(feedback_history) > 0,
            "feedback_text": " ".join(feedback_history) if feedback_history else "",
        }
        
        # Call transformer router
        try:
            result = infer_transformer_router(
                ckpt_path=router_ckpt_path,
                state=state,
                candidates=retrieval_pool,
                prompt_sims=prompt_sims,
                reflection_sims=reflection_sims,
                negative_penalties=negative_penalties,
            )
            router_mix = result["router_mix"]
            router_conf = result["router_conf"]
        except Exception as e:
            print(f"[warn] Router inference failed: {e}")
            return None, [0.0, 0.0, 0.0], 0.0
        
        # If confidence too low, return None (fallback to reflexion)
        if router_conf < router_conf_threshold:
            return None, router_mix, router_conf
        
        # Compute weighted scores for each candidate
        weighted_scores = []
        for i, traj in enumerate(retrieval_pool):
            score = (
                router_mix[0] * prompt_sims[i] +
                router_mix[1] * reflection_sims[i] -
                router_mix[2] * negative_penalties[i]
            )
            weighted_scores.append(score)
        
        # Select best candidate
        best_idx = max(range(len(weighted_scores)), key=lambda i: weighted_scores[i])
        selected_traj = retrieval_pool[best_idx]
        
        return selected_traj, router_mix, router_conf'''
    
    content = re.sub(ensemble_pattern, ensemble_replacement, content, flags=re.DOTALL)
    
    # 3. Update the call site to use new function
    call_pattern = r'select_stage2_traj_ensemble\('
    call_replacement = 'select_stage2_traj_transformer_router('
    content = re.sub(call_pattern, call_replacement, content)
    
    # Write back
    with open('paramAgent.py', 'w') as f:
        f.write(content)
    
    print("Router integration completed!")
    return True

if __name__ == "__main__":
    try:
        changed = integrate_router()
        sys.exit(0 if changed else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)
