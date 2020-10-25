# The Minion Game 
import re

def minion_game(string):
    len_str = len(string)
    number_all_substring = (len_str + 1)*len_str//2
    indeces_vowels = [m.start() for m in re.finditer("['A','E','I','O','U']",string)]
    
    countVow = len_str * len(indeces_vowels) - sum(indeces_vowels)
    
    if countVow > number_all_substring//2:
        print('Kevin', countVow)
    elif countVow == number_all_substring//2:
        print('Draw')
    else:
        print("Stuart", number_all_substring - countVow)

if __name__ == '__main__':
    s = input()    
    minion_game(s)


# Athlete Sort
from operator import itemgetter
if __name__ == '__main__':
    
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

    for x in arr:
        x.append(arr.index(x))
    
    result = sorted(arr, key = itemgetter(k,m))
    
    for y in result:
        print(*y[0:m])


# No Idea!
if __name__ == '__main__':
    
    n, m = input().split()
    arr = list(map(int, input().split()))
    like = set(map(int, input().split()))
    dontlike = set(map(int, input().split()))
    
    results = [ 1 if x in like else -1 if x in dontlike else 0 for x in arr]
    
    print(sum(results))

#Zipped!
if __name__ == '__main__':
    
    nStudents, xSubjects = list(map( int , input().split()))
    subjectsMarks = [tuple(map(float, input().split())) for _ in range(xSubjects)]    
    students = list(zip(*subjectsMarks))   

    for student in students:
       print('{0:.1f}'.format((sum(student)/xSubjects)))

#Python Evaluation
if __name__ == '__main__':
    var = input()
    eval(var)


#Any or All
def palidrom(stringa):
    result = True
    for i in range(len(stringa)//2):
        result = result and stringa[i] == stringa[len(stringa) - i -1]
    return result

if __name__ == '__main__':
    
    n = int(input())
    numbers = input().split()    
    checkPositive = [int(x) > 0 for x in numbers]

    if (all(checkPositive) == False):
        print('False')
        exit()
    else:
        checkPalindrom = [palidrom(y) for y in numbers]
        print(any(checkPalindrom))


# Tuple
if __name__ == '__main__':
    n = int(input())
    integer_list = list(map(int, input().split()))
    t = tuple(integer_list)    
    print(hash(t))


# Lists
if __name__ == '__main__':
    N = int(input())
    lista = []
    for _ in range(N):
        comand = input().split()
        if (comand[0] == 'append'):
            lista.append(int(comand[1]))
        if (comand[0] == 'remove'):
            lista.remove(int(comand[1]))
        if (comand[0] == 'insert'):
            lista.insert(int(comand[1]),int(comand[2]))
        if(comand[0] == 'print'):
            print(lista)
        if(comand[0] == 'sort'):
            lista.sort()
        if(comand[0] == 'pop'):
            lista.pop()
        if (comand[0] == 'reverse'):
            lista.reverse()
        

# Arrays
import numpy
def arrays(arr):    
    arr.reverse()
    return numpy.array(arr,float)  
arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape
import numpy as np
nums = input().strip().split()
a = np.array(nums, int)
print(np.reshape(a, (3,3)))

# Transpose and Flatten
import numpy as np
n, m = map(int, input().split())
mat = [list(map(int, input().split())) for _ in range(n)]
matrice = np.array(mat, int)
print(np.transpose(matrice))
print(matrice.flatten())

# Concatenate
import numpy as np
n, m, p = map(int, input().split())
listA = [list(map(int, input().split())) for _ in range(n)]
listB = [list(map(int, input().split())) for _ in range(m)]

matA = np.array(listA, int)
matB = np.array(listB,int)
matC = np.concatenate((matA, matB))
print(matC)

# Zeros and Ones 
import numpy as np
params = list(map(int, input().split()))
z = np.zeros((params),dtype=int) 
o  = np.ones((params),dtype=int) 
print(z)
print(o)


# Eye and Identity
import numpy as np
n,m = list(map(int,input().split()))
e = str(np.eye(n,m,k = 0)).replace('0',' 0').replace('1',' 1')
print(e)


#Array Mathematics
import numpy as np
n,m = list(map(int,input().split()))
aList = [list(map(int,input().split())) for _ in range(n)]
bList = [list(map(int,input().split())) for _ in range(n)]

a = np.array(aList)
b = np.array(bList)

print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

#Floor, Ceil and Rint
import numpy as np
a = np.array(list(map(float,input().split())))
print(str(np.floor(a)).replace('[', '[ ').replace('.','. ').replace(' ]',']'))
print(str(np.ceil(a)).replace('[', '[ ').replace('.','. ').replace(' ]',']'))
print(str(np.rint(a)).replace('[', '[ ').replace('.','. ').replace(' ]',']'))


#Sum and Prod
import numpy as np
n , m = map(int,input().split())
items = [[list(map(int,input().split()))] for _ in range(n)]
a = np.array(items, dtype=int)
somme = a.sum(axis=0)
print(somme.prod())

#Min and Max
import numpy as np
n , m = map(int,input().split())
items = [list(map(int,input().split())) for _ in range(n)]
a = np.array(items, dtype=int)
minimi = np.min(a, axis = 1)
print(np.max(minimi))


#Mean, Var and Std
import numpy as np
n , m = map(int,input().split())
items = [list(map(int,input().split())) for _ in range(n)]
a = np.array(items, dtype=int)
np.set_printoptions(legacy='1.13')
print(np.mean(a,axis=1))
print(np.var(a,axis=0))
print(np.around(np.std(a),12))

#Dot and Cross
import numpy as np
n = int(input())
a =np.array( [list(map(int, input().split())) for _ in range(n)], dtype=int)
b =np.array( [list(map(int, input().split())) for _ in range(n)], dtype=int)
print(np.dot(a,b))

#Inner and Outer
import numpy as np 
a = np.array(list(map(int,input().split())) )
b = np.array(list(map(int,input().split())) )
print(np.inner(a,b))
print(np.outer(a,b))

#Polynomials
import numpy as np 
coeffs = list(map(float, input().split()))
x = float(input())
print(np.polyval(coeffs, x))

#Linear Algebra
import numpy as np 
n = int(input())
a =np.array( [list(map(float, input().split())) for _ in range(n)], dtype=float)
det = np.linalg.det(a)
print(np.round(det,2))

#Group(), Groups() & Groupdict()
import re
s = input()
m = re.search(r'([a-zA-Z0-9])\1',s)
print(m.group(1))


#Re.findall() & Re.finditer()
import re
s = input()
m = re.findall("(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIOU]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])", s)
if not m:
    print("-1")
else:
    for e in m:
        print(e)  

# Re.split()
regex_pattern = r"[,/.]"	# Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))

# Re.start() & Re.end()
import re 
s = input()
pat = input()

start = 0
if re.search(pat, s):
    while start + len(pat) < len(s):
        m = re.search(pat,s[start:])
        if m:
            print("({0}, {1})".format(start + m.start(), start + m.end()-1))
            start += m.start() + 1
        else: 
            exit()
else:
    print("(-1, -1)")

# Regex Substitution
import re
n = int(input())
s = [input() for _ in range(n)]
a = "((?<=\s)\&\&(?=\s))"
#b = "\s\|\|\s"
b = "((?<=\s)\|\|(?=\s))"

for line in s:
    res = re.sub(a, "and", line)
    resFin = re.sub(b, "or", res)
    print(resFin)

#Validating Roman Numerals
import re
t = 'M{0,3}'
h = '(C[MD]|D?C{0,3})'
x = '(X[CL]|L?X{0,3})'
d = '(I[VX]|V?I{0,3})$'
regex_pattern = t+h+x+d
print(str(bool(re.match(regex_pattern, input()))))

# Validating and Parsing Email Addresses
from email.utils import parseaddr
from email.utils import formataddr
import re
n = int(input())

for _ in range(n):
    s = input()
    name, email = parseaddr(s)    

    m = re.match("[A-Za-z](\w|\-|\.|\_)*@[A-Za-z]+\.([A-Za-z]+$)", email)
    if (m):
        
        if (len(m.group(2)) > 0 and len(m.group(2)) < 4):
            print(formataddr((name,email)))

#Hex Color Code
import re
n = int(input())
for _ in range(n):
    lines = input()    
    matches = re.findall("(?<!^)(#(?:[0-9a-fA-F]{3}){1,2})",lines)
    for i in matches:
        print(i)


#Validating Credit Card Numbers
import re
def consecutive(s):
    count = 1
    s = s.replace('-','')
    for i in range(1,len(s)):
        if s[i-1] == s[i]:
            count += 1
        else: 
            count=1
        if count == 4:
            return False
    return True

n = int(input())
for _ in range(n):
    s = input()
    isNotConsecutive = consecutive(s) 
    correctLen = r"^[456][0-9]{3}(-?)[0-9]{4}\1[0-9]{4}\1[0-9]{4}"
    m = re.fullmatch(correctLen,s)
    isCorrectLen = bool(m)

    if isCorrectLen and isNotConsecutive:
        print('Valid')
    else:
        print('Invalid')


#Matrix Script
import re

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

result = list(map(list, zip(*matrix)))
total_string = ""
for item in result:
    for c in item:
        total_string += c

print(re.sub(r"(?<=(\w))([^a-zA-Z0-9_]|[\s])+(?=(\w))", ' ',total_string))


# HTML Parser - Part 1
from html.parser import HTMLParser

class htmlParserA(HTMLParser):
    def handle_starttag(self,tag,attrs):
        print("Start :", tag)
        for attr in attrs:
            print("->",attr[0],">",attr[1])
    def handle_endtag(self,tag):
        print("End   :", tag)
    def handle_startendtag(self,tag,attrs):
        print("Empty :", tag)
        for attr in attrs:
            print("->",attr[0],">",attr[1])

myparser = htmlParserA()
n = int(input())
text = ""
for _ in range(n):
    text += input() 

myparser.feed(text)



#HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        lines_data = data.split('\n')
        if not lines_data: return
        if (len(lines_data) == 1):
            print(">>> Single-line Comment")            
            print(lines_data[0])
        else:
            print(">>> Multi-line Comment")
            for line in lines_data:
                if line:
                    print(line)
            
    def handle_data(self,data):
        lines_data = data.split('\n')
        if not any(lines_data): return
        print(">>> Data")
        for line in lines_data:
            if line:
                print(line)
     
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()



# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser

class htmlParserA(HTMLParser):
    def handle_starttag(self,tag,attrs):
        print(tag)
        for attr in attrs:
            print("->",attr[0],">",attr[1])
    
    def handle_startendtag(self,tag,attrs):
        print(tag)
        for attr in attrs:
            print("->",attr[0],">",attr[1])

myparser = htmlParserA()
n = int(input())
text = ""
for _ in range(n):
    text += input() 

myparser.feed(text)


#Validating UID
import re

n = int(input())
for _ in range(n):
    uid = input()

    rightNumber = r"[A-Za-z0-9]{10}"
    reapet = r"(\w+)\w*\1"
    atLeast3d = r"([A-Za-z]*\d){3,}"
    atLeast2Upper = r"([A-Z][A-Za-z0-9]*){2,}"

    flagRightNumber = bool(re.match(rightNumber,uid))
    flagReapet = bool( bool(re.search(reapet,uid)) == False)
    flag3D = bool(re.search(atLeast3d,uid))
    flag2U = bool(re.search(atLeast2Upper,uid))

    if (flagRightNumber and flagReapet and flag3D and flag2U):
        print("Valid")
    else:
        print("Invalid")




#XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here    
    return len(node.attrib) + sum(get_attr_number(item) for item in node)
    
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))



#XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1        
    for child in elem:
        depth(child, level + 1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)



#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        new_l = []
        for item in l:
            start = len(item) - 10
            new_l.append('+91 ' + item[start :start+5] + ' ' + item [start + 5:len(item)])      
        return f(new_l)       
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 



#Decorators 2 - Name Directory
import operator

def person_lister(f):
    def inner(people):
        for person in people:
            person[2] = int(person[2])       
        sortedPeople = sorted( people, key = operator.itemgetter(2))
        sortedPeople = [f(p) for p in sortedPeople]
        return sortedPeople
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# Introduction to Sets
def average(array):
    plan = set(array)
    return sum(plan)/len(plan)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)



# Symmetric Difference
m = int(input())
a = set(map(int,input().split()))
n = int(input())
b = set(map(int,input().split()))

inters = a.intersection(b)
u = a.union(b)
result = u.difference(inters)
for item in sorted(result):
    print(item)


# Set.add()
n = int(input())
countries = set()
for _ in range(n):
    countries.add(input())
print(len(countries))


# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

o = int(input())
for _ in range(o):
    line = input()
    if line == "pop":
        s.pop()
    else: 
        comand , param = line.split()
        param = int(param)
        try:
            if comand == "remove":
                s.remove(param)
        except KeyError:
            pass           
        if comand == "discard":
            s.discard(param)

print(sum(s))


# Set .union() Operation
n = int(input())
a = set(input().split())
m = int(input())
b = set(input().split())
print(len(a.union(b)))


# Set .intersection() Operation
n = int(input())
a = set(input().split())
m = int(input())
b = set(input().split())
print(len(a.intersection(b)))

# Set .difference() Operation
n = int(input())
a = set(input().split())
m = int(input())
b = set(input().split())
print(len(a.difference(b)))

# Set .symmetric_difference() Operation
n = int(input())
a = set(input().split())
m = int(input())
b = set(input().split())
u = a.union(b)
inter = a.intersection(b)
print(len(u.difference(inter)))



# Set Mutations
lenA = int(input())
A = set(map(int,input().split()))
num_of_set = int(input())
for _ in range(num_of_set):
    comand, m = input().split()
    B = set(map(int, input().split()))
    if comand == "intersection_update":
        A.intersection_update(B)
    if comand == "symmetric_difference_update":
        A.symmetric_difference_update(B)
    if comand == "difference_update":
        A.difference_update(B)
    if comand == "update":
        A.update(B)
print(sum(A))


# The Captain's Room
k = int(input())
L = list(map(int, input().split()))
S = set(L)
print((sum(S)*k - sum(L))//(k-1))


# Check Subset
n = int(input())
for _ in range(n):
    na = int(input())
    A = set(map(int, input().split()))
    nb = int(input())
    B = set(map(int, input().split()))
    print (len(A.intersection(B)) == len(A))


# Check Strict Superset
A = set(map(int, input().split()))
n = int(input())
check = []
for _ in range(n):
    B = set(map(int,input().split()))
    check.append((B.issubset(A)) and (len(A.difference(B)) != 0))
print(all(check))

# Swap Case
def swap_case(s):
    return s.swapcase()
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


# String Split and Join
def split_and_join(line):
    # write your code here
    splitted = line.split()
    return "-".join(splitted)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#What's Your Name?
def print_full_name(a, b):
    print("Hello",a,b + "!","You just delved into python.")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


# Mutations
def mutate_string(string, position, character):
    lista = list(string)
    lista[position] = character
    return ''.join(lista)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


# Find a string
def count_substring(string, sub_string):
    count=start=0
    while True:
        start = string.find(sub_string, start) + 1
        if start > 0:
            count += 1
        else:
            return count    

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()    
    count = count_substring(string, sub_string)
    print(count)


# String Validators
if __name__ == '__main__':
    s = input()    
    print (any(c.isalnum()  for c in s))
    print (any(c.isalpha() for c in s))
    print (any(c.isdigit() for c in s))
    print (any(c.islower() for c in s))
    print (any(c.isupper() for c in s))


# ginortS
import string
print(*sorted(input(), key=(string.ascii_letters + '1357902468').index), sep='')


#collections.Counter()
x = input()
shoe_sizes =list( map(int, input().split()))
n = int(input())
shoes = [list(map(int, input().split())) for _ in range(n)]
earned = 0
for shoe in shoes:
    if shoe[0] in shoe_sizes:
        earned += shoe[1]
        shoe_sizes.remove(shoe[0])
print(earned)


# DefaultDict Tutorial
from collections import defaultdict
n, m = input().split()
dizio = defaultdict(list)
for i in range(int(n)):
    word = input()
    dizio[word].append(i+1)
for j in range(int(m)):
    target = input()
    print(*dizio[target] or [-1])


# Collections.deque()
from collections import deque
d = deque()
for _ in range(int(input())):
    comand = input()
    if comand == "pop":
        d.pop()
    elif comand == "popleft":
        d.popleft()
    else:
        c, param = comand.split()
        if c == "append":
            d.append(param)
        if c == "appendleft":
            d.appendleft(param)
print(*d)


# Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    s = []
    if n == 0:
        return s
    if n == 1:
        s.append(0)
        return s
    else: 
        s = [0,1]
        for i in range(2,n):
            s.append(s[i-2]+s[i-1])        
        return s

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))



# Validating Email Addresses With a Filter
import re
def fun(s):
    return bool(re.fullmatch(r"([_A-Za-z0-9-]+)@([A-Za-z0-9]+)\.(\w){1,3}",s))

def filter_mail(emails):
    return list(filter(fun, emails))

if __name__ == '__main__':
    n = int(input())
    emails = []
    for _ in range(n):
        emails.append(input())

filtered_emails = filter_mail(emails)
filtered_emails.sort()
print(filtered_emails)


# Reduce Function
from fractions import Fraction
from functools import reduce
import operator

def product(fracs):
    t = reduce(operator.mul , fracs)
    return t.numerator, t.denominator

if __name__ == '__main__':
    fracs = []
    for _ in range(int(input())):
        fracs.append(Fraction(*map(int, input().split())))
    result = product(fracs)
    print(*result)


# Calendar Module
import calendar
mm,dd,aaaa=map(int,input().split())
print((calendar.day_name[calendar.weekday(aaaa,mm,dd)]).upper())


# Text Wrap
import textwrap

def wrap(string, max_width):
    chunks = [ string[start:start + max_width] for start in range(0, len(string), max_width)]
    return '\n'.join(chunks)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


# Designer Door Mat
n, m = map(int, input().split())

for i in range(n):
    rombo = ".|."
    if i < (n-1)/2:
        print((rombo * (2*i+1)).center(m, "-"))
    elif i == (n-1)/2:
        print("WELCOME".center(m, "-"))
    else:
        print((rombo * (2*(n-1-i)+1)).center(m, "-"))


# String Formatting
def print_formatted(number):
    c = len(bin(number)) - 2
    for i in range(1, number + 1):
        print('{0:{c}d} {0:{c}o} {0:{c}X} {0:{c}b}'.format(i, c=c))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)



# Alphabet Rangoli
import string
def print_rangoli(size):
    larghezza = (size - 1)*4 + 1 #17
    altezza = 2*size - 1     
    
    lettere = string.ascii_lowercase
    for i in range(size-1,-size,-1):
        temp = '-'.join(lettere[size-1:abs(i):-1]+lettere[abs(i):size])
        print(temp.center(4*size-3,'-'))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!
import re
# Complete the solve function below.
def solve(s):    
    lista = [i.capitalize() for i in s.split(' ')]
    return ' '.join(lista)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()




# Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    try:
        k = (x2 - x1)/ (v1-v2)
        if (k > 0 and k.is_integer()):
            return "YES"
        else:
            return "NO"
    except:
        return "NO"

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# Viral Advertising

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
    people = 5 
    cumulative = 0
    for i in range(n):           
        people = int(math.floor(people/2))
        cumulative += people
        people *= 3
     
    return cumulative

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

def superDigit(n, k):    
    lista = re.findall(r"\d",n)
    num = sum(list(map(int, lista)))
    num *= k
    while (num > 10):
        lista = re.findall(r"\d",str(num))
        num = sum(list(map(int, lista)))
    return num

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)
    print(result)
    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1
#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    to_sort = arr[len(arr) -1]
    idx = len(arr) - 2
    for i in range(idx, -1, -1):
        if (arr[i] <= to_sort):
            break
        arr[i + 1] = arr[i]
        print(*arr)
    arr[i+1] = to_sort
    print(*arr)
        
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1,n):
        item = arr[i]
        idx = i -1
        while idx >= 0 and arr[idx] > item:
            arr[idx + 1] = arr[idx] 
            idx -= 1
        arr[idx +1] = item
        print(*arr)
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)




# Birthday Cake Candles
import sys
import os
def birthdayCakeCandles(candles):    
    return(candles.count(max(candles)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()