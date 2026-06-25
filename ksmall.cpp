#include <bits/stdc++.h>
using namespace std;
#define ll long long
int main() {
  int t;
  cin>>t;
  while(t--)
  {
     priority_queue<int> q;
      int n,k,temp;
      cin>>n>>k;
      for(int i=0;i<n;i++)
      {
          cin>>temp;
          q.push(temp);
      }
      for(int i=1;i<k;i++)
      {
          q.pop();
      }
      cout<<q.top()<<endl;
  }
}
