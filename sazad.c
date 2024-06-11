#include <stdio.h>
#include <stdlib.h>

#define SIZE 5  // Define the size of the queue

typedef struct {
    int items[SIZE];
    int front;
    int rear;
} Queue;

// Function to create a new queue
Queue* createQueue() {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->front = -1;
    q->rear = -1;
    return q;
}

// Function to check if the queue is full
int isFull(Queue* q) {
    if ((q->front == 0 && q->rear == SIZE - 1) || (q->front == q->rear + 1)) {
        return 1;
    }
    return 0;
}

// Function to check if the queue is empty
int isEmpty(Queue* q) {
    if (q->front == -1) return 1;
    return 0;
}

// Function to add an element to the queue
void enqueue(Queue* q, int value) {
    if (isFull(q)) {
        printf("Queue is full\n");
        return;
    }
    if (q->front == -1) {
        q->front = 0;
    }
    q->rear = (q->rear + 1) % SIZE;
    q->items[q->rear] = value;
    printf("Inserted %d\n", value);
}

// Function to remove an element from the queue
int dequeue(Queue* q) {
    int element;
    if (isEmpty(q)) {
        printf("Queue is empty\n");
        return -1;
    } else {
        element = q->items[q->front];
        if (q->front == q->rear) {
            q->front = -1;
            q->rear = -1;
        } else {
            q->front = (q->front + 1) % SIZE;
        }
        printf("Removed %d\n", element);
        return element;
    }
}

// Function to display the elements of the queue
void display(Queue* q) {
    int i;
    if (isEmpty(q)) {
        printf("Queue is empty\n");
    } else {
        printf("Queue elements are:\n");
        for (i = q->front; i != q->rear; i = (i + 1) % SIZE) {
            printf("%d ", q->items[i]);
        }
        printf("%d\n", q->items[i]);
    }
}

// Main function to test the queue
int main() {
    Queue* q = createQueue();

    enqueue(q, 10);
    enqueue(q, 20);
    enqueue(q, 30);
    enqueue(q, 40);
    enqueue(q, 50);

    display(q);

    dequeue(q);
    display(q);

    enqueue(q, 60);
    display(q);

    return 0;
}
