#include <bits/stdc++.h>

using namespace std;

class BinaryTreeNode {
public:
    int data;
    BinaryTreeNode* left;
    BinaryTreeNode* right;

    BinaryTreeNode(int node_data) {
        data = node_data;
        left = right = NULL;
    }
};

BinaryTreeNode* insert_node_into_binary_tree(BinaryTreeNode* root, int node_data) {
    if (!root) {
        root = new BinaryTreeNode(node_data);
    } else {
        if (node_data <= root->data) {
            root->left = insert_node_into_binary_tree(root->left, node_data);
        } else {
            root->right = insert_node_into_binary_tree(root->right, node_data);
        }
    }

    return root;
}

/*
 * For your reference:
 *
 * BinaryTreeNode {
 *      data;
 *     BinaryTreeNode* left;
 *     BinaryTreeNode* right;
 * };
 *
 */

int maxDepth(BinaryTreeNode* root) {
    
    if(!root){
        return 0;
    }
    int left = maxDepth(root->left);
    int right = maxDepth(root->right);
    return max(left,right) + 1;
}

int main()
{
    int n;
    cin>>n;

    BinaryTreeNode* root = NULL;

    for (int i = 0; i < n; i++) {
        int x;
        cin>>x;
        root = insert_node_into_binary_tree(root, x);
    }
    int ans = maxDepth(root);
  	cout<<ans;
    return 0;
}
