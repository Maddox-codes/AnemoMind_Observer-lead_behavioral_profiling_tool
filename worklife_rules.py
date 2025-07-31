def get_work_life_compatibility(anxiety_score, stability_classification, integrity_score, observer_gut_feeling):
    """
    Determines work-life compatibility based on a set of rules.

    Args:
        anxiety_score (float): The predicted anxiety score (0-10).
        stability_classification (str): 'Stable' or 'Unstable'.
        integrity_score (float): The predicted integrity score (0-100).
        observer_gut_feeling (str): 'Good', 'Neutral', or 'Bad'.

    Returns:
        tuple: A tuple containing the compatibility label and a summary of the reasoning.
    """
    reasons = []

    # Evaluate Anxiety
    if anxiety_score >= 7:
        reasons.append("high stress levels")
    
    # Evaluate Stability
    if stability_classification == 'Unstable':
        reasons.append("cognitive instability")

    # Evaluate Integrity
    if integrity_score < 40:
        reasons.append("very low integrity score")
    elif integrity_score < 60:
        reasons.append("low integrity score")

    # Evaluate Gut Feeling
    if observer_gut_feeling == 'Bad':
        reasons.append("negative observer gut feeling")

    # Determine final recommendation
    if stability_classification == 'Unstable' or integrity_score < 40 or observer_gut_feeling == 'Bad':
        if len(reasons) > 1 :
            summary = "multiple concerning signals: " + ", ".join(reasons)
        else:
            summary = "critical concern regarding " + reasons[0]
        return "❌ Low Fit", summary
        
    if len(reasons) >= 2:
        return "⚠️ Moderate Fit", "Combination of factors: " + ", ".join(reasons)
    
    if len(reasons) == 1:
        return "⚠️ Moderate Fit", "One area of concern: " + reasons[0]

    return "✅ Good Fit", "All indicators are within acceptable ranges."