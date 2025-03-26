//get the size of the linkedlist in java
class Hello
{
    class Node
    {
        String data;
        Node next;
        Node(String data)
        {
            this.data=data;
            this.next=null;
        }
    }
    Node head;
    public void addlast(String data)
    {
        Node newnode =new Node(data);
        if(head==null)
        {
            head=newnode;
            return;
        }
        Node current=head;
        while(current.next!=null)
        {
            current=current.next;
        }
        current.next=newnode;
    }
    public void printsize()
    {
        int size=0;
        Node current=head;
        if(head==null)
        {
            System.out.println("null");
            return;
        }
        while(current!=null)
        {
            current=current.next;
            size++;
        }
        System.out.println(size);
    }
    public static void main(String args[])
    {
        Hello obj=new Hello();
        obj.printsize();
        obj.addlast("manish ");
        obj.addlast("is");
        obj.addlast("good boy");
        obj.printsize();

    }

}
