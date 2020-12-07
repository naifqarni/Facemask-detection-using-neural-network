
# import all components
# from the tkinter library
from tkinter import *

# import filedialog module
from tkinter import filedialog
import os
import mask_from_image
import mask_from_video
import webbrowser
import model_trainer
from tkinter import ttk
import graphs

# Function for opening the
# file explorer root


def detect_image():
    filename = filedialog.askopenfilename(initialdir="/",title="Select a File",filetypes=(("images files","*.jpg*"),("all files","*.*")))

    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    print(filename)
    if filename != None and filename != '':
        mask_from_image.image_Detection(filename)


def webcam():
    mask_from_video.webcam(0)


def vid():
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(("images files", "*.mp4*"),("video files","*.*")))

    # if a file not selected do not do any thing
    if filename != None and filename != '':
        mask_from_video.webcam(filename)


def train_and_wait(b_size, epocc, l_rate):
    model_trainer.train(int(b_size.get()), int(
        epocc.get()), float(l_rate.get()))


def train():
    filewin = Toplevel(root)
    filewin.geometry("300x300")
    warn = Label(
        filewin, text='this will take a while, \n the gui will not respond until the traning is done')
    warn.pack()
    e1 = Entry(filewin)
    e1.insert(0, 'enter batch size')
    e1.pack()
    e2 = Entry(filewin)
    e2.insert(1, 'enter number of epoch')
    e2.pack()
    e3 = Entry(filewin)
    e3.insert(2, 'enter learning rate')
    e3.pack()
    btn = Button(filewin,
                 text="Start Traing!",
                 command=lambda: train_and_wait(e1, e2, e3))
    btn.pack()

def test_conf_mat():
    filename = filedialog.askdirectory()

    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    print(filename)
    if filename != None and filename != '':
        graphs.test_and_graph(filename)

def stat():
    filewin = Toplevel(root)
    filewin.geometry("300x300")
    btn1 = Button(filewin,
                 text="open test diroctry to predict and show accuracy",
                 command=test_conf_mat )
    btn2 = Button(filewin,
                 text="show train acuracy and loss",
                 command=lambda:graphs.train_acuracy_loss())
    btn3 = Button(filewin,
                 text="show validation data acuracy and loss",
                 command=lambda:graphs.val_acc_loss())
    btn1.pack()
    btn2.pack()
    btn3.pack()

# Create the root root
root = Tk()
root.iconbitmap('images/file_type_ai_icon_130757.ico')

# Set root title
root.title('Face Detection')

# Set root size
root.geometry("700x400")

# Set root background color
root.config(background="white")

# Create a File Explorer label
label_file_explorer = Label(root,
                            text="Face detection using Opencv and keras",
                            width=99, height=4, bg='white',
                            )


button_explore = Button(root,
                        text="Detect from an image",
                        command=detect_image, width=33, height=4)


button_webcam = Button(root,
                       text="webcam",
                       command=webcam, width=33, height=4)
button_vid = Button(root,
                    text="video",
                    command=vid, width=33, height=4)
button_Train = Button(root,
                      text="Train    ",
                      command=train, width=99, height=4)
button_stat = Button(root,
                     text="Statistics",
                     command=stat, width=99, height=4)


# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column=0, row=1, columnspan=3)

button_explore.grid(column=0, row=2, columnspan=1)

button_webcam.grid(column=1, row=2, columnspan=1)
button_vid.grid(column=2, row=2, columnspan=1)
button_Train.grid(column=0, row=3, columnspan=3)
button_stat.grid(column=0, row=4, columnspan=3)
# Let the root wait for any events


def about():
    filewin = Toplevel(root)
    filewin.geometry("300x300")
    label = Label(filewin, text='ics 381 project \n by naif and ahmed')
    label.pack()


menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)


filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
editmenu = Menu(menubar, tearoff=0)


editmenu.add_separator()


def github():
    webbrowser.open("https://github.com/naifqar/face_mask_detector")


helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="github page", command=github)
helpmenu.add_command(label="About...", command=about)
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)


root.mainloop()
