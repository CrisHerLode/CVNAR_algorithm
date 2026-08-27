def calcMetric(data, supports):
    """
    Metrics from antecedent/consequent/rule supports.

    Index map (kept stable for main_cvoa callers):
      0 conf, 1 lift, 2 leverage_norm, 3 accuracy, 4 support, 5 cf, 6 cf2,
      7 leverage (P(AC)-P(A)P(C)), 8 accuracy2, 9 gain, 10 wracc, 11 conviction,
      12 netconf, 13 yule_q

    Formulas (N = |data|):
      conf = P(C|A) = rule/ant
      lift = conf / P(C)
      leverage = P(AC) - P(A)P(C)
      gain = conf - P(C)
      WRAcc = P(A) * (conf - P(C))   # classic Lavrac; equals leverage
      conviction = (1-P(C)) / (1-conf) if conf < 1 else +inf
      netconf = (P(AC)-P(A)P(C)) / (P(A)*(1-P(A)))
      Yule's Q = (ad-bc)/(ad+bc) with a=AC, b=A-AC, c=C-AC, d=N-A-C+AC
    """
    metrics = []
    support_ant = supports[0]
    support_cons = supports[1]
    support_rule = supports[2]
    n = len(data.index)
    p_cons = support_cons / n if n else 0.0

    if support_ant != 0:
        conf = support_rule / support_ant
        if conf > p_cons:
            cf = ((support_rule * n) - (support_ant * support_cons)) / ((n - support_cons) * support_ant) if (n - support_cons) else 0.0
            cf2 = (conf - p_cons) / (1 - p_cons) if p_cons < 1 else 0.0
        else:
            if support_cons != 0:
                cf = ((support_rule * n) - (support_ant * support_cons)) / (support_ant * support_cons)
                cf2 = (conf - p_cons) / p_cons
            else:
                cf = 0
                cf2 = 0
    else:
        conf = 0
        cf = 0
        cf2 = 0

    if support_cons != 0:
        lift = conf * n / support_cons
        gain = conf - p_cons
    else:
        lift = 0
        gain = 0

    leverage = ((support_rule * n) - (support_ant * support_cons)) / pow(n, 2) if n else 0.0
    leverage_norm = (leverage + 0.25) / 0.5
    # Alias of classic leverage (kept for backward compatibility with "Leverage metric 2")
    leverage_plain = (support_rule / n) - ((support_ant / n) * (support_cons / n)) if n else 0.0
    accuracy = (support_rule + (n - (support_ant + support_cons - support_rule))) / n if n else 0.0
    accuracy2 = (support_rule / n) + (1 - ((support_ant / n) + (support_cons / n) - (support_rule / n))) if n else 0.0
    support = support_rule / n if n else 0.0

    # WRAcc (Lavrac): P(A)*(conf - P(C)) == leverage_plain
    p_ant = support_ant / n if n else 0.0
    wracc = p_ant * (conf - p_cons) if n else 0.0

    if conf >= 1.0 - 1e-15:
        conviction = float("inf") if (1.0 - p_cons) > 0 else 0.0
    else:
        conviction = (1.0 - p_cons) / (1.0 - conf)

    # Netconf (common NAR form used in VLMOHSNAR-style tables)
    den_net = p_ant * (1.0 - p_ant)
    if n and den_net > 1e-12:
        netconf = (leverage_plain) / den_net
    else:
        netconf = 0.0

    # Yule's Q from 2x2 contingency
    a = float(support_rule)
    b = float(support_ant - support_rule)
    c = float(support_cons - support_rule)
    d = float(n - support_ant - support_cons + support_rule) if n else 0.0
    a = max(a, 0.0)
    b = max(b, 0.0)
    c = max(c, 0.0)
    d = max(d, 0.0)
    den_yule = a * d + b * c
    yule_q = ((a * d - b * c) / den_yule) if den_yule > 0 else 0.0

    metrics.append(conf)
    metrics.append(lift)
    metrics.append(leverage_norm)
    metrics.append(accuracy)
    metrics.append(support)
    metrics.append(cf)
    metrics.append(cf2)
    metrics.append(leverage_plain)
    metrics.append(accuracy2)
    metrics.append(gain)
    metrics.append(wracc)
    metrics.append(conviction)
    metrics.append(netconf)
    metrics.append(yule_q)

    return metrics


def metrics_from_supports(support_ant, support_cons, support_rule, n_rows):
    """Derive extra metrics from supports (for tops without reloading CSV)."""
    if not n_rows:
        return {
            "gain": None,
            "leverage": None,
            "wracc": None,
            "conviction": None,
            "netconf": None,
            "yule_q": None,
        }
    class _Dummy:
        index = range(n_rows)

    m = calcMetric(_Dummy(), [support_ant, support_cons, support_rule])
    return {
        "gain": m[9],
        "leverage": m[7],
        "wracc": m[10],
        "conviction": m[11],
        "netconf": m[12],
        "yule_q": m[13],
    }
