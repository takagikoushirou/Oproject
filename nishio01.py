n=2000
c=0
for i in range(1,n+1):
    for j in range(1,i+1):
        p=j/i
        if p>=0.2725 and p<0.2735:
            c+=1
print(str(c)+'通り')