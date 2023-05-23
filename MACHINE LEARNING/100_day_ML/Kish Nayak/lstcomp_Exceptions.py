#################################### LIST FORMATING ####################################

lst=[x*x for x in [1,2,3,4,5,6,7,8,9,10]]
print(lst)


last1=[x*x for x in  [1,2,3,4,5,6,7,8,9,10] if x%2==0 ]
print(last1)



#################################### Python Exception Handling ####################################

try:
    ##code block where exception can occur
    a=1
    b="s"
    c=a+b
except NameError as ex1:
    print("The user have not defined the variable")
except Exception as ex:
    print(ex)
    
    

### try else
try:
    ##code block where exception can occur
    a=int(input("Enter the number 1 "))
    b=int(input("Enter the number 2 "))
    c=a/b 
    d=a*b
    e=a+b
    
except NameError:
    print("The user have not defined the variable")
except ZeroDivisionError:
    print("Please provide number greater than 0")
except TypeError:
    print("Try to make the datatype similar")
except Exception as ex:
    print(ex)
else:
    print(c)
    print(d)
    print(e)
    
    
### try else
try:
    ##code block where exception can occur
    a=int(input("Enter the number 1 "))
    b=int(input("Enter the number 2 "))
    c=a/b
    
except NameError:
    print("The user have not defined the variable")
except ZeroDivisionError:
    print("Please provide number greater than 0")
except TypeError:
    print("Try to make the datatype similar")
except Exception as ex:
    print(ex)
else:                                           #if the exceptions are ok the execute this
    print(c)
finally:                                        #this will be executed no matter what
    print("The execution is done")
    
    
#########################################################################
class Error(Exception):
    pass

year=int(input("Enter your DOB:"))
age=2023-year
try:
    if 20<=age<=30:
        print("Yor age is valid and you can write the exam")
    else:
        raise Error
except Error:
    print("You are not in between the age limit")


    
#########################################################################
class Error(Exception):
    pass

year=int(input("Enter your DOB:"))
age=2023-year

try:
    if 20<age<30:
        print('ok')
    else:
        raise Error
except Error:
    print("not valid")
else:
    print('ok2')
finally:
    print('done')
