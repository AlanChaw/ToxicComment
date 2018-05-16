import enchant
import re
from autocorrect import spell

d = enchant.Dict("en_US")


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


word = "shiiiiit"
word = reduce_lengthening(word)
print(word)
# # use the autocorrect tool
# print(spell(word))


# use the enchant tool
print(d.check(word))
print(d.suggest(word))



# 03d6e5da188d5e16,youuuuuuuuuuuuuuuuuuuuuuuuuuu alllllllllllllllllllllllllllllllllllllllll lllllllllllllllllllllllooooooooooooooooooooooooooooooooooooooookkkkkkkkkkkkkkkkkkkkkkkkkk llllllllllllllllliiiiiiiiiiiiiiiiiiiiiiiiiikkkkkkkkkkkkkkkkkkkkkkkkkkkeeeeeeeeeeeeeeeeeeeee sssssssssssssssssshhhhhhhhhhhhhhhhhhhhhhhhhhiiiiiiiiiiiiiiiiiiiiiiiiiiitttttttttttttttt
