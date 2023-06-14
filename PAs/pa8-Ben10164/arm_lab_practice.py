import itertools

header = ["level", "lang", "tweets", "phd", "interviewed_well"]
table = [
    ["Senior", "Java", "no", "no", "False"],
    ["Senior", "Java", "no", "yes", "False"],
    ["Mid", "Python", "no", "no", "True"],
    ["Junior", "Python", "no", "no", "True"],
    ["Junior", "R", "yes", "no", "True"],
    ["Junior", "R", "yes", "yes", "False"],
    ["Mid", "R", "yes", "yes", "True"],
    ["Senior", "Python", "no", "no", "False"],
    ["Senior", "R", "yes", "no", "True"],
    ["Junior", "Python", "yes", "no", "True"],
    ["Senior", "Python", "yes", "yes", "True"],
    ["Mid", "Python", "no", "yes", "True"],
    ["Mid", "Java", "yes", "no", "True"],
    ["Junior", "Python", "no", "yes", "False"],
]


def prepend_attribute_label(table, header):
    for row in table:
        for i in range(len(row)):
            row[i] = header[i] + "=" + str(row[i])


prepend_attribute_label(table, header)

rule1 = {"lhs": ["interviewed_well=False"], "rhs": ["tweets=no"]}
# task: create rule5
rule5 = {"lhs": ["phd=no", "tweets=yes"], "rhs": ["interviewed_well=True"]}

# utility function


def check_row_match(terms, row):
    # return 1 if all the terms are in the row
    # 0 otherwise
    for term in terms:
        if term not in row:
            return 0
    return 1


# lab task #2


def compute_rule_counts(rule, table):
    Nleft = Nright = Nboth = Ntotal = 0
    for row in table:
        Nleft += check_row_match(rule["lhs"], row)
        Nright += check_row_match(rule["rhs"], row)
        Nboth += check_row_match(rule["lhs"] + rule["rhs"], row)
        Ntotal += 1
    return Nleft, Nright, Nboth, Ntotal


Nleft, Nright, Nboth, Ntotal = compute_rule_counts(rule1, table)
print(Nleft, Nright, Nboth, Ntotal)

# lab task #3


def compute_rule_interestingness(rule, table):
    Nleft, Nright, Nboth, Ntotal = compute_rule_counts(rule, table)

    rule["confidence"] = Nboth / Nleft
    rule["support"] = Nboth / Ntotal
    rule["completeness"] = Nboth / Nright
    # NOTE: denominators could be 0


for rule in [rule1, rule5]:
    compute_rule_interestingness(rule, table)
    print(rule)
