import cv2,os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sqlite3

(im_width, im_height) = (312, 312)

enroll=raw_input('Check the list of students enrolled Y/N ')
if(enroll=="Y"):
    print "list of students enroled \n"
    conn = sqlite3.connect('FaceBase.db')
    cursor = conn.execute("SELECT id, name  from people")
    for row in cursor:
        print "ID = ", row[0]
        print "NAME = ", row[1], "\n"
    conn.close()
    print "";



cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.createLBPHFaceRecognizer()
def insertorUpdate(Id1,Name):
    conn=sqlite3.connect("FaceBase.db")
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS people (id unique, name text)')
    cmd="SELECT * FROM people WHERE ID="+str(Id1)
    cursor=conn.execute(cmd)
    isRecordExit=0
    for row in cursor:
        isRecordExit=1
    if(isRecordExit==1):
        c.execute("UPDATE people SET Name='"+str(Name)+"' WHERE ID="+str(Id1))
    else:
        c.execute("INSERT INTO people VALUES("+str(Id1)+", '"+str(Name)+"')")
    conn.commit()
    conn.close()




en=raw_input('Are you Enrolled Y/N ')
Id1=""
if(en=="N"):
    Id1=raw_input('enter your id ')
    Name=raw_input('enter your name ')
    insertorUpdate(Id1,Name)
    sampleNum=0
    while(True):
        ret, img = cam.read();
        gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        #gray1=img.convert('L')
        faces = detector.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #incrementing sample number 
            sampleNum=sampleNum+1
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            #saving the captured face in the dataset folder
            cv2.imwrite("dataSet/User."+Id1 +'.'+ str(sampleNum) + ".jpg", face_resize)
            cv2.imshow('frame',img)
        #wait for 100 miliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        #break if the sample number is more than 20
        elif sampleNum>20:
            break
else:
    Id1=raw_input('enter your id ')

    
cam.release()
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces,np.array(Ids))
recognizer.save('trainner/trainningset.yml')

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner\\trainningset.yml')
def getProfile(Id2):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM people WHERE ID="+str(Id2)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

path='image1'
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #print "hjbkjb"
        x1=np.array(Image.open(imagePath),'uint8')
        img1 = Image.open(imagePath)
        color = [255, 0, 0]
        
        imageNp=np.array(pilImage,'uint8')
        faces=detector.detectMultiScale(imageNp,1.3,5)
        for(x,y,w,h) in faces:
            face = imageNp[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            Id2, conf = recognizer.predict(face_resize)
            dr = ImageDraw.Draw(img1)
            dr.rectangle(((x,y),(x+w,y+h)), fill="black", outline = "blue")
            #print "sbkkjbs"
            """if(conf<50):
                if(Id==1):
                    n='Brajesh'
            else:
                n='Unknown'"""
            print Id1
            print Id2
            print conf
            n='';
            Id3=int(Id1)
            #if(Id1==Id):
            if(Id3==Id2):
                profile=getProfile(Id2)
                n= (profile[1])
            else:
                n='Unknown'
            print n
            mypath = 'E:\FaceRecognition\i'
            mypath=mypath + n
            if not os.path.exists(mypath):
               os.makedirs(mypath)
            filename = os.path.split(imagePath)[-1]
            Image.open(imagePath).save(mypath+'/'+filename)
            mypath = 'E:\FaceRecognition\FaceRecognized'
            if not os.path.exists(mypath):
                os.makedirs(mypath)
            filename = os.path.split(imagePath)[-1]
            img1.save(mypath+'/'+filename)
            Id=0

cv2.destroyAllWindows()

