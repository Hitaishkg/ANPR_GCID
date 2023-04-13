str='vshnuv'
x={}
k=1
num=''
for i in str:
    x[i]=0
for i in str:
    x[i]=x[i]+1
for i in str:
    if(x[i]==1):
        num=num+i
print(num)