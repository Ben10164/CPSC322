import numpy as np

from mysklearn.myruleminer import MyAssociationRuleMiner

# note: order is actual/received student value, expected/solution


def test_association_rule_miner_fit():

    # toy market basket analysis dataset
    transactions = [
        ["b", "c", "m"],
        ["b", "c", "e", "m", "s"],
        ["b"],
        ["c", "e", "s"],
        ["c"],
        ["b", "c", "s"],
        ["c", "e", "s"],
        ["c", "e"],
    ]

    # this is from the apriori lab
    # 4. rules for S = {b,c,m} with minconf = 80%
    # confidence = 100% (2/2)
    apriori_rule1 = {"lhs": ["c", "m"], "rhs": ["b"]}
    # confidence = 100% (2/2)
    apriori_rule2 = {"lhs": ["b", "m"], "rhs": ["c"]}
    apriori_rule3 = {"lhs": ["b", "c"], "rhs": ["m"]}  # confidence = 66% (2/3)
    apriori_rule4 = {"lhs": ["m"], "rhs": ["b", "c"]}  # confidence = 100% (2/2)
    apriori_rule5 = {"lhs": ["c"], "rhs": ["b", "m"]}  # confidence = 28% (2/7)
    apriori_rule6 = {"lhs": ["b"], "rhs": ["c", "m"]}  # confidence = 50% (2/4)

    # create the association rule miner (we used minsup = 25% and minconf = 80% in the lab)
    apriori_arm = MyAssociationRuleMiner(0.25, 0.8)
    apriori_arm.fit(transactions)
    apriori_arm_rules = apriori_arm.rules

    apriori_arm_rules_actual = [
        {"lhs": "m", "rhs": "b", "confidence": 1.0, "support": 0.25},
        {"lhs": "m", "rhs": "c", "confidence": 1.0, "support": 0.25},
        {"lhs": "e", "rhs": "c", "confidence": 1.0, "support": 0.5},
        {"lhs": "s", "rhs": "c", "confidence": 1.0, "support": 0.5},
        {"lhs": "cm", "rhs": "b", "confidence": 1.0, "support": 0.25},
        {"lhs": "es", "rhs": "c", "confidence": 1.0, "support": 0.375},
    ]
    assert apriori_arm_rules == apriori_arm_rules_actual
