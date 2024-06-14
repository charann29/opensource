amount=[10000,15000,20000,50000,100000]
won=0
questions=["1:The International Literacy Day is observed on","2:The language of Lakshadweep. a Union Territory of India, is","3:In which group of places the Kumbha Mela is held every twelve years?","4:Bahubali festival is related to","5:Which day is observed as the World Standards  Day?"]
options=[['Sep 8','Nov 28','May 2','Sep 22'],['Tamil','Hindu','Malayalam','Telugu'],['Ujjain. Purl; Prayag. Haridwar','Prayag. Haridwar, Ujjain,. Nasik','Rameshwaram. Purl, Badrinath. Dwarika','Chittakoot, Ujjain, Prayag,Haridwar'],['Islam','Hinduism','Buddism','Jainism'],['June 26','Oct 14','Nov 15','Dec 2']]
correctoptions=[1,3,2,4,2]
print("welcome to kbc")
print("you will be having 5 questions costing from 10000 to 100000")
print("if you choose correct ans you will move on to next question else you will return home with what you have")
for i in range(5):
    print("Your No {} question is".format(i+1))
    print(questions[i])
    print("A:{}        B:{}\nC:{}        D:{}".format(options[i][0],options[i][1],options[i][2],options[i][3]))
    choose=int(input("enter your option in interger format:"))
    if choose<1 or choose>4:
        print('entered option is invalid\nplease enter an interger between 1-4')
        choose=int(input("enter your option in interger format:"))
    if(choose==correctoptions[i]):
        print("hooray you have won:",amount[i])
        won+=amount[i]
    else:
        print("sorry")
        print('you can go home with',won)
        break
        
        
