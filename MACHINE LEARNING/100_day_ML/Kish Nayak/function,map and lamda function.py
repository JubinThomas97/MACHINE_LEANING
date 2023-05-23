######################## Function ########################

def oddORevenSum(lst):
    odd_sum=0
    even_sum=0
    for i in lst:
        if i%2==0:
            even_sum=even_sum+i
        else :
            odd_sum=odd_sum+i
    return f"ODD sum ={odd_sum} \nEVEN sum={even_sum}"

ans=oddORevenSum([1,2,3,4,5,6,7,8,9,10,45,4523,15,23])
print(ans)


######################## MAP Function ########################

def ret_odd_even(lst):
    for i in lst:
        if i % 2==0:
            return f"The number {i} is EVEN"
        else:
            return f"The number {i} is ODD"
lst1=[1,2,3,45,82,545132,1965,165312,165,784,131,383,794]
ans1=ret_odd_even(lst1)
print(ans1)
        
# this cant go through every items in the list so here we use mapping function

def ret_odd_even(num):
    if num % 2==0:
        return f"The number {num} is EVEN"
    else:
        return f"The number {num} is ODD"
    
lst1=[1,2,3,45,82,545132,1965,165312,165,784,131,383,794]
ans=map(ret_odd_even,lst1)
print(list(ans))


######################## LAMDA Function ########################
# Anonymous function , a function with no name

ans2=lambda a,b:a+b
ans2(10,20)

ans3=lambda c:c%2==0
ans3(3)

addition=lambda x,y,z:x+y+z
addition(5,10,20)



######################## FILTER Function ########################


def even_numbers(num):
    if num%2==0:
        return True
lst=[1,2,3,4,5,6] 
list(filter(even_numbers,lst))

#or
list(filter(lambda num:num%2==0,lst))               # [2, 4, 6]     

list(map(lambda num:num%2==0,lst)) #[False, True, False, True, False, True]


