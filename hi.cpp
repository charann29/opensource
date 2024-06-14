#include <bits/stdc++.h>

using namespace std;

int main() {
    int a[12]={-1},b[12],val,i=0;
    while(cin>>val){
    a[i]=val;
        i++;
    }
    int low=0;
    for(int j=0;j<i;j++)
      if(a[j]<a[low])
      low=j;
      int k=0;
      for(int j=0;j<i;j++){
      if(j==low)
      continue;
      b[k]=a[j];
      k++;
      }
     for(int w=0;w<k-1;w++)
     {
         int v,lcm,a1=a[w],b=a[w+1];
        if(a1>b)
        lcm = a1;
        else
        lcm = b;
        while(1) {
        if( lcm%a1==0 && lcm%b==0 ) {
         v=lcm;
         break;}
      lcm++;
        } 
         a[w+1]=v;
     }
     int ans=a[k-1];
     int y=0;
     for(int j=0;j<i-1;j++)
     {
         int t=0;
      int pornot=ans+a[low];
      for(int u=2;u<=sqrt(pornot);u++)
      if(pornot%u==0)
      {
          t=1;
          break;
      }
      if(t==0){
      cout<<pornot;y=1;
          break;
      }
     }
    if(y==0)
    cout<<"None";
      
}