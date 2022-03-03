"""
1 5 yes
2 6 yes
1 5 no
1 5 no
1 6 yes
2 6 no
1 5 yes
1 6 yes
"""

data = [[1, 5], [2, 6], [1, 5], [1, 5], [1, 6], [2, 6], [1, 5], [1, 6]]
result = ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes']

# 1.
# p(result = yes)
num_yes = result.count('yes')
# p(result = no)
num_no = result.count('no')

print('p(result = yes) = {}'.format(num_yes / len(result)))
print('p(result = no) = {}'.format(num_no / len(result)))

assert num_yes + num_no == len(result)

# 2.
# attribute 1 can be either 1 or two (col 0)
# attribute 2 can be either 5 or 6 (col 1)
# now we can find the probability of the result being yes or no

# prob of attribute 1 given that result is yes
# p(att1 = 1 | result = yes)
# we can expand this to
# p(att=1 and result = yes) / p(result = yes)
numerator = 0
for i in range(len(data)):
    if data[i][0] == 1 and result[i] == 'yes':
        numerator += 1
print(numerator)
denometator = num_yes  # we already did this :)
result = numerator / denometator
print('p(att1 = 1 | result = yes) = {}'.format(result))


# 3. p(result=yes|X) = p(X|result=yes)*P*result=yes) = p(att1=1|result=yes)p(att2=5|result=yes)p(result=yes)
# p(att1=1|result=yes) = 4/5
# p(att2=5|result=yes) = 2/5
# p(result=yes) = 5/8
# = 1/5

# p(result=no|X) = p(X|result=no)*P*result=no) = p(att1=1|result=no)p(att2=5|result=no)p(result=no)
# p(att1=1|result=no) = 2/3
# p(att2=5|result=no) = 2/3
# p(result=no) = 3/8
# = 1/6

# since we have p(result=yes) = 1/5 and p(result=no) = 1/6
# and 1/5 > 1/6, we can predict that the result is yes


# 4. p(result=yes|X)
# = (p(att1=1 and att2 = 5 and result=yes) * p(result=yes))/p(result=yes)
# = p(att1=1 and att2 = 5 and result=yes)
# = 2/8 = 1/4

# p(result=no|X)
# = (p(att1=1 and att2 = 5 and result=no) * p(result=no))/p(result=no)
# there are 2 that are 1 5 no, out of the 8, so that means that:
# p(att1=1 and att2 = 5 and result=no) = 2/8
# and the p(result=no)s cancel out, so:
# p(result=no|X) = 1/4

# since we have p(result=yes|X) = p(result=no|X) = 1/4,
# we chose the class label with the larger prior, which is:
# p(result=yes) = 5/8
# therefore the result is yes!

# ON TIE, WE PICK THE CLASS WITH THE LARGER PRIOR
