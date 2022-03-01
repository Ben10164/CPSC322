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

# 4. p(result=yes|X)
# = p(att1=1 and att2 = 5 and result=yes) * p(result=yes)/p(result=yes)
# = p(att1=1 and att2 = 5 and result=yes)
# = 2/8 = 1/4
