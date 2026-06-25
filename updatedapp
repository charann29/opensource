from flask import Flask, render_template, request, jsonify
import csv

app = Flask(__name__)

# File path to the CSV
CSV_FILE = "books.csv"

# Helper function to read data from CSV
def read_csv():
    with open(CSV_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

# Helper function to write data to CSV
def write_csv(data):
    try:
        with open(CSV_FILE, mode='w', newline='') as file:
            fieldnames = ['id', 'title', 'author', 'genre']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    except Exception as e:
        print("Error writing to CSV:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(read_csv())

@app.route('/books', methods=['POST'])
def add_book():
    data = read_csv()
    new_id = max(int(book['id']) for book in data) + 1 if data else 1
    new_book = {
        'id': str(new_id),
        'title': request.json['title'],
        'author': request.json['author'],
        'genre': request.json['genre']
    }
    data.append(new_book)
    write_csv(data)
    return jsonify(new_book)

@app.route('/books/<int:id>', methods=['PUT'])
def update_book(id):
    data = read_csv()
    updated_book = None
    for book in data:
        if int(book['id']) == id:
            book['title'] = request.json['title']
            book['author'] = request.json['author']
            book['genre'] = request.json['genre']
            updated_book = book
            break
    if updated_book:
        write_csv(data)
        return jsonify(updated_book)
    else:
        return jsonify({"message": "Book not found"}), 404

@app.route('/books/<int:id>', methods=['DELETE'])
def delete_book(id):
    data = read_csv()
    original_length = len(data)
    data = [book for book in data if int(book['id']) != id]
    if len(data) < original_length:
        write_csv(data)
        return jsonify({"message": "Book deleted successfully"})
    else:
        return jsonify({"message": "Book not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
