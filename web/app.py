import numpy as np
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect,send_from_directory
import tensorflow as tf
import wget,io,math
import os,cv2,base64
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import matplotlib
from tensorflow import keras
import numpy as np
from numpy import expand_dims

matplotlib.use('Agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

#import pickle
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS1={'ckpt','pb','h5'}

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_model(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS1



@app.route('/opss',methods=['GET','POST'])
def opss():    
    full_filename = os.path.join('static','cat.jpg')
    print(full_filename)
    model = tf.keras.models.load_model('op75.ckpt')
    return render_template("layer_click.html", user_image = full_filename)

model_loaded=""


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if os.path.exists(UPLOAD_FOLDER):
            for files in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, files)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
        else:
            os.mkdir(UPLOAD_FOLDER)
        if 'image' not in request.files:
            flash('Please select a file','warning')
            return redirect(request.url)
        if 'model' not in request.files:
            flash('No file part','error')
            return render_template("Home.html")
        file = request.files['image']
        model = request.files['model']
        modename = request.form.get('modelname')
        # if user does not select file, browser also
        # submit an empty part without filename
        print(1)
        a=0
        b=0
        if file.filename == '':
            flash('Please select a file','warning')
            a=1
            # return redirect(request.url)
        if model.filename == '':
            flash('Please select a file','error')
            b=1
        if a==1 or b==1:
            return redirect(request.url)
        a=0
        b=0
        if allowed_file(file.filename)==False:
            flash('Please select the file in vailid format','warning')
            a=1
        if allowed_model(model.filename)==False:
            flash('Please select the file in vailid format','error')
            b=1
        if a==1 or b==1:
            return redirect(request.url)
        a=0
        b=0
        x=""
        # file is name of image
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imagename=filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            a=1
            print(filename)
        if model and allowed_model(model.filename):
            filename = secure_filename(model.filename)
            x=filename
            modelname=filename
            model.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            b=1
            print(filename)
        if a==1 and b==1:
            print(a,b)
            return redirect(url_for('prediction',filename=x))
    return render_template("Home.html")

@app.route('/uploads/<filename>',methods=['GET', 'POST'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/prediction/<string:filename>',methods=['GET','POST'])
def prediction(filename):
    file_path=os.path.join(UPLOAD_FOLDER,filename)
    model = tf.keras.models.load_model(file_path)
    ops=[]
    model.summary(print_fn=lambda x: ops.append(x))
    dic={}
    dic['heading']=['S.No.','Layer (type)','Output Shape','Param #','Connected to']
    heading=ops[2]
    dic['para']=ops[-4:-1]
    dic['model_name']='Model Name'
    list=ops[4:-4]
    a=[]
    for x in list:
        if x[0]=='_' or x[0]=='=':
            continue
        else:
            a.append(x.split())

    layer_name=[]
    para=[]
    output_shape=[]
    connection=[]
    for y in a:
        if len(y)<2:
            continue
        s=""
        p=""
        c=""
        s+=y[0]
        idx=2
        for i in range(0,len(y)):
            if y[i]=='(None,':
                idx=i
                break
        s+=y[idx-1]
        par_idx=0
        for i in range(idx,len(y)-1):
            x=y[i]
            p+=x
            if ')' in x:
                par_idx=i+1
                break
        if ')' not in s:
            s+=')'
        para.append(y[par_idx])
        layer_name.append(s)
        output_shape.append(p)
        for i in range(par_idx+1,len(y)):
            c+=y[i]
        connection.append(c)

    dic['layer_name']=layer_name
    dic['para_counts']=para
    dic['output_shape']=output_shape
    dic['connection']=connection
    dic['count']=len(layer_name)
    print(len(layer_name))
    xx='model.png'
    keras.utils.plot_model(model,to_file='static/{}'.format(xx),show_shapes=True)
    return render_template('submit.html',summary=ops,dic=dic,count=len(layer_name),xx=xx)

    
    

@app.route('/about',methods=['GET','POST'])
def about():
    return render_template('about.html',title='About')


@app.route('/show',methods=['GET','POST'])
def show():
    list=[]
    return render_template('layer_click.html',list=list)


@app.route('/layer_click/<string:name>/<int:id>',methods=['GET','POST'])
def layer_click(name,id):
    # imgg=cv2.imread('static/ops.jpg',cv2.IMREAD_COLOR)
    # model = tf.keras.models.load_model('op75.ckpt')
    # plt.imshow(img)
    # plt.savefig('static/test.png')
    imagename=""
    modelname=""
    just_name=""
    for files in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, files)
        if allowed_file(file_path):
            just_name=files.rsplit('.',1)[0]
            imagename=file_path
        else:
            modelname=file_path
    ops=[]
    model = tf.keras.models.load_model(modelname)
    model.summary(print_fn=lambda x: ops.append(x))
    # print(modelname)
    # print(imagename)
    # print(len(ops))
    # print(just_name)
    for files in os.listdir('static'):
        file_path = os.path.join('static', files)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    x=model.layers[0].get_input_at(0).get_shape().as_list()[1]
    y=model.layers[0].get_input_at(0).get_shape().as_list()[2]
    z=model.layers[0].get_input_at(0).get_shape().as_list()[3]
    channel=1
    if z==1:
        channel=0
    img=cv2.imread(imagename,channel)
    imgg=cv2.resize(img,dsize=(x,y))
    img=imgg
    img=np.array(img)
    img = expand_dims(img, axis=0)
    if channel==0:
        img=expand_dims(img, -1)
    layer_outputs = [model.layers[id].output]
    new_model=tf.keras.Model(inputs=model.inputs,outputs=layer_outputs)
    feature_maps = new_model.predict(img)
    l=len(feature_maps[0].shape)
    if l<3:
        return render_template('layer_click.html',list=[],ops="error",name=name,out_shape=feature_maps[0].shape)
    ix=1
    zz=feature_maps[0].shape[2]
    square=math.floor(zz)
    list=[]
    tmp=zz
    print(tmp)
    ss=1
    tit=1
    while(True):
        ix=1
        fig=plt.figure()
        for _ in range(9):
            if tit>zz:
                break
            ax=fig.add_subplot(3, 3, ix)
            ax.title.set_text('channel {}'.format(tit))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0][:,:,tit-1])
            ix+=1
            tmp-=1
            tit+=1
        just_name+=name
        picName='static/{}{}.png'.format(just_name,ss)
        plt.savefig(picName)
        # full_filename = os.path.join('static','cat.jpg')
        org='{}{}.png'.format(just_name,ss)
        list.append(org)
        ss+=1
        if tit>zz:
            break
    return render_template('layer_click.html',list=list,name=name,out_shape=feature_maps[0].shape)



# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     #prediction = model.predict([np.array(list(data.values()))])

#     output = 1
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)