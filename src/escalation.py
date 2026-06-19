def should_escalate(query, results):

    sensitive_keywords = [
        "billing",
        "refund",
        "legal",
        "lawsuit",
        "account ownership"
    ]

    if len(results) == 0:
        return True

    query = query.lower()

    for keyword in sensitive_keywords:
        if keyword in query:
            return True

    return False