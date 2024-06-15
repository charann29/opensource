# Function to add a new user
def add_user(name, email):
    conn = sqlite3.connect('library.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
    conn.commit()
    conn.close()

# Function to issue a book to a user
def issue_book(user_id, book_id):
    conn = sqlite3.connect('library.db')
    cursor = conn.cursor()
    cursor.execute('SELECT copies FROM books WHERE id = ?', (book_id,))
    copies = cursor.fetchone()[0]
    if copies > 0:
        cursor.execute('UPDATE books SET copies = copies - 1 WHERE id = ?', (book_id,))
        cursor.execute('INSERT INTO transactions (user_id, book_id, issue_date) VALUES (?, ?, ?)', 
                       (user_id, book_id, datetime.now().strftime('%Y-%m-%d')))
    else:
        print('No copies available.')
    conn.commit()
    conn.close()

# Function to return a book
def return_book(transaction_id):
    conn = sqlite3.connect('library.db')
    cursor = conn.cursor()
    cursor.execute('SELECT book_id FROM transactions WHERE id = ?', (transaction_id,))
    book_id = cursor.fetchone()[0]
    cursor.execute('UPDATE books SET copies = copies + 1 WHERE id = ?', (book_id,))
    cursor.execute('UPDATE transactions SET return_date = ? WHERE id = ?', 
                   (datetime.now().strftime('%Y-%m-%d'), transaction_id))
    conn.commit()
    conn.close()
def main():
    while True:
        print("\nLibrary Management System")
        print("1. Add Book")
        print("2. Add User")
        print("3. Issue Book")
        print("4. Return Book")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            title = input("Enter book title: ")
            author = input("Enter book author: ")
            isbn = input("Enter book ISBN: ")
            copies = int(input("Enter number of copies: "))
            add_book(title, author, isbn, copies)
        elif choice == '2':
            name = input("Enter user name: ")
            email = input("Enter user email: ")
            add_user(name, email)
        elif choice == '3':
            user_id = int(input("Enter user ID: "))
            book_id = int(input("Enter book ID: "))
            issue_book(user_id, book_id)
        elif choice == '4':
            transaction_id = int(input("Enter transaction ID: "))
            return_book(transaction_id)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
