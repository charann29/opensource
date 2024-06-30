from flask import Flask, render_template, request, redirect, url_for, session
import pymysql
from datetime import datetime

app = Flask(__name__)

app.secret_key = 'welcome'
global uname


@app.route('/Deposit', methods=['GET', 'POST'])
def Deposit():
    output = '<tr><td><font size="3" color="black">Username</td><td><input type="text" name="t1" size="20" value='+uname+' readonly/></td></tr>'
    return render_template('Deposit.html', msg1=output)

@app.route('/Withdraw', methods=['GET', 'POST'])
def Withdraw():
    output = '<tr><td><font size="3" color="black">Username</td><td><input type="text" name="t1" size="20" value='+uname+' readonly/></td></tr>'
    return render_template('Withdraw.html', msg1=output)

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')


@app.route('/Login', methods=['GET', 'POST'])
def Login():
   return render_template('Login.html', msg='')

@app.route('/Signup', methods=['GET', 'POST'])
def Signup():
    return render_template('Signup.html', msg='')


@app.route('/ViewBalance', methods=['GET', 'POST'])
def ViewBalance():
    font = "<font size='3' color='black'>" 
    output = ""
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM transaction where username='"+uname+"'")
        rows = cur.fetchall()
        for row in rows:
            output+="<tr><td>"+font+str(row[0])+"</font></td>"
            output+="<td>"+font+str(row[1])+"</font></td>"
            output+="<td>"+font+str(row[2])+"</font></td>"
            output+="<td>"+font+str(row[3])+"</font></td>"
            output+="<td>"+font+str(row[4])+"</font></td>"
    return render_template('ViewBalance.html', msg=output)


@app.route('/LoginAction', methods=['GET', 'POST'])
def LoginAction():
    global uname
    if request.method == 'POST':
        user = request.form['t1']
        password = request.form['t2']
        data = request.files['t3'].read() 
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM users")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user and password == row[1]:
                    in_file = open("E:/Fingerprintbasedatmbatch2/Fingerprintbasedatm"+user+".png", "rb")
                    avail_data = in_file.read() # if you only wanted to read 512 bytes, do .read(512)
                    in_file.close()
                    if avail_data == data:
                        index = 1
                        uname = user
                        break		
        if index == 0:
            return render_template('Login.html', msg="Invalid login details")
        else:
            return render_template('UserScreen.html', msg="Welcome "+uname)

        
def getAmount(user):
    amount = 0
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM transaction")
        rows = cur.fetchall()
        for row in rows:
            if row[0] == user:
                amount = float(row[4])
                break
    return amount

@app.route('/WithdrawAction', methods=['GET', 'POST'])
def WithdrawAction():
    if request.method == 'POST':
        user = request.form['t1']
        amount = request.form['t2']
        total = getAmount(user)
        withdraw =  float(amount)
        status = "Error in depositing amount"
        if total > withdraw:
            total = total - withdraw
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_sql_query = "update transaction set transaction_amount='"+amount+"',transaction_type='Withdrawl',transaction_date='"+str(timestamp)+"',total_balance='"+str(total)+"' where username='"+user+"'"
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
            db_cursor = db_connection.cursor()
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            if db_cursor.rowcount == 1:
                status = "Withdrawl Transaction Successfull"
        else:
            status = "Insufficient Fund"
        return render_template('UserScreen.html', msg=status)    
    


@app.route('/DepositAction', methods=['GET', 'POST'])
def DepositAction():
    if request.method == 'POST':
        user = request.form['t1']
        amount = request.form['t2']
        total = getAmount(user)
        status = "Error in depositing amount"
        if total == 0:
            total = total + float(amount)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_sql_query = "INSERT INTO transaction(username,transaction_amount,transaction_type,transaction_date,total_balance) VALUES('"+user+"','"+amount+"','Deposit','"+str(timestamp)+"','"+str(total)+"')"
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
            db_cursor = db_connection.cursor()
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            if db_cursor.rowcount == 1:
                status = "Transaction Successfull"
        elif total > 0:
            total = total + float(amount)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            student_sql_query = "update transaction set transaction_amount='"+amount+"',transaction_type='Deposit',transaction_date='"+str(timestamp)+"',total_balance='"+str(total)+"' where username='"+user+"'"
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
            db_cursor = db_connection.cursor()
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            if db_cursor.rowcount == 1:
                status = "Transaction Successfull"
        return render_template('UserScreen.html', msg=status)    

@app.route('/SignupAction', methods=['GET', 'POST'])
def SignupAction():
    if request.method == 'POST':
        user = request.form['t1']
        password = request.form['t2']
        phone = request.form['t3']
        email = request.form['t4']
        address = request.form['t5']
        gender = request.form['t6']
        data = request.files['t7'].read()  
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM users")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user:
                    status = user+" Username already exists"
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'atm',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO users(username,password,contact_no,emailid,address,gender) VALUES('"+user+"','"+password+"','"+phone+"','"+email+"','"+address+"','"+gender+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            if db_cursor.rowcount == 1:
                out_file = open("E:/Fingerprintbasedatmbatch2/Fingerprintbasedatm"+user+".png", "wb")
                out_file.write(data)
                out_file.close()
                status = 'Signup process completed'
        return render_template('Signup.html', msg=status)    
        
        


@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')



if __name__ == '__main__':
    app.run()










