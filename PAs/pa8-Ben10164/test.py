import numpy as np

from mysklearn.myruleminer import MyAssociationRuleMiner

# note: order is actual/received student value, expected/solution


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
apriori_rule1 = {"lhs": ["c", "m"], "rhs": ["b"]}  # confidence = 100% (2/2)
apriori_rule2 = {"lhs": ["b", "m"], "rhs": ["c"]}  # confidence = 100% (2/2)
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


print("apriori_arm_rules:")
print(apriori_arm_rules)
assert apriori_arm_rules == apriori_arm_rules_actual

apriori_arm.print_association_rules()

# # interview dataset
# header = ["level", "lang", "tweets", "phd", "interviewed_well"]
# table = [
#     ["Senior", "Java", "no", "no", "False"],
#     ["Senior", "Java", "no", "yes", "False"],
#     ["Mid", "Python", "no", "no", "True"],
#     ["Junior", "Python", "no", "no", "True"],
#     ["Junior", "R", "yes", "no", "True"],
#     ["Junior", "R", "yes", "yes", "False"],
#     ["Mid", "R", "yes", "yes", "True"],
#     ["Senior", "Python", "no", "no", "False"],
#     ["Senior", "R", "yes", "no", "True"],
#     ["Junior", "Python", "yes", "no", "True"],
#     ["Senior", "Python", "yes", "yes", "True"],
#     ["Mid", "Python", "no", "yes", "True"],
#     ["Mid", "Java", "yes", "no", "True"],
#     ["Junior", "Python", "no", "yes", "False"],
# ]
# interview_rule1 = {"lhs": ["att4=False"], "rhs": ["att2=no"]}
# interview_rule5 = {
#     "lhs": ["att3=no", "att2=yes"],
#     "rhs": ["att4=True"],
# }
# interview_arm = (
#     MyAssociationRuleMiner()
# )  # i forgot what the values of minsup and minconf were
# interview_arm.fit(table)
# interview_arm_rules = interview_arm.rules
# interview_rules_actual = []
# interview_rules_actual.append(
#     {
#         "lhs": ["att4=False"],
#         "rhs": ["att2=no"],
#         "confidence": 0.8,
#         "support": 0.2857142857142857,
#         "completeness": 0.5714285714285714,
#     }
# )
# interview_rules_actual.append(
#     {
#         "lhs": ["att3=no", "att2=yes"],
#         "rhs": ["att4=True"],
#         "confidence": 1.0,
#         "support": 0.2857142857142857,
#         "completeness": 0.4444444444444444,
#     }
# )
