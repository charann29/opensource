#include <bits/stdc++.h>
using namespace std;

int n;
string s;
vector<char>st;
vector<char>a;

void standinput()
{
  for(auto x:st)
  {
     cout<<x<<" ";
  }
  for(auto z:a)
  {
    cout<<z<<" ";
  }
}

void check()
{
    for(int i=0;i<st.size();i++)
    {
       if(st[i]=='a')
       {
          st[i]='E';
          standinput();
          cout<<"REduced E->A";
          check();
       }
       else if(i+2<st.size()&&st[i]=='E'&&st[i+1]=='+'||st[i+1]=='*'||st[i]=='E')
       {
           st.pop_back();
           st.pop_back();
            standinput();
           if(st[i+1]=='+')
           {
             cout<<"reduce E->E+E"<<endl;
           }
           else if(st[i+1]=='*')
           {
             cout<<"reduce E->E8E"<<endl;
           }
       }
    }
}

int main()
{
    cin>>s;
    for(int i=0;i<s.size();i++){
        cin>>a[i];
    }
    

    return 0;
}