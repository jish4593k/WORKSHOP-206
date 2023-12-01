import socket
from concurrent.futures import ThreadPoolExecutor
import random
import turtle
from tkinter import Tk, Label, Button
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import tensorflow as tf

class QuizServer:
    def __init__(self, ip_address, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((ip_address, port))
        self.server.listen()
        self.list_of_clients = []
        self.questions = [
            # ... (same questions as in the original code)
        ]
        self.answers = ['d', 'a', 'b', 'a', 'a', 'a', 'a', 'b', 'a', 'c', 'b', 'd', 'd', 'c', 'a', 'b', 'a']

    def get_random_question_answer(self, conn):
        random_index = random.randint(0, len(self.questions) - 1)
        random_question = self.questions[random_index]
        random_answer = self.answers[random_index]
        conn.send(random_question.encode('utf-8'))
        return random_index, random_question, random_answer

    def remove_question(self, index):
        self.questions.pop(index)
        self.answers.pop(index)

    def client_thread(self, conn):
        score = 0
        conn.send("Welcome to this quiz game!".encode('utf-8'))
        conn.send("You will receive a question. The answer to that question should be one of a, b, c or d\n".encode('utf-8'))
        conn.send("Good Luck!\n\n".encode('utf-8'))
        index, question, answer = self.get_random_question_answer(conn)
        while True:
            try:
                message = conn.recv(2048).decode('utf-8')
                if message:
                    if message.lower() == answer:
                        score += 1
                        conn.send(f"Bravo! Your score is {score}\n\n".encode('utf-8'))
                    else:
                        conn.send("Incorrect answer! Better luck next time!\n\n".encode('utf-8'))
                    self.remove_question(index)
                    index, question, answer = self.get_random_question_answer(conn)
                else:
                    self.remove(conn)
            except Exception as e:
                print(f"Error: {e}")
                continue

    def remove(self, connection):
        if connection in self.list_of_clients:
            self.list_of_clients.remove(connection)

    def run_server(self):
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                conn, addr = self.server.accept()
                self.list_of_clients.append(conn)
                print(addr[0] + " connected")
                executor.submit(self.client_thread, conn)

def tensor_operations_example():

    tensor_example = tf.constant([[1, 2], [3, 4]])
    tensor_squared = tf.square(tensor_example)
    print("Tensor Squared:")
    print(tensor_squared)

def turtle_graphics_example():

    turtle.forward(100)
    turtle.right(90)
    turtle.forward(100)
    turtle.done()

def gui_example():

    root = Tk()
    label = Label(root, text="Hello, GUI!")
    button = Button(root, text="Click me")
    label.pack()
    button.pack()
    root.mainloop()

def sklearn_example():
 
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

def keras_example():

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=8))
    model.add(Dense(units=1, activation='sigmoid'))

def data_mining_example():

    data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [28, 35, 22]}
    df = pd.DataFrame(data)

def data_processing_example():

    array = np.array([[1, 2, 3], [4, 5, 6]])
    sum_result = np.sum(array)
    print("Sum of the array:", sum_result)

if __name__ == "__main__":
    # Initialize and run the quiz server
    quiz_server = QuizServer(ip_address='127.0.0.1', port=8000)
    quiz_server.run_server()

    # Run additional examples
    tensor_operations_example()
    turtle_graphics_example()
    gui_example()
    sklearn_example()
    keras_example()
    data_mining_example()
    data_processing_example()
