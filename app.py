from flask import Flask, send_file, render_template, request
from scipy import misc
from api import posterize_numpy, numpy_to_img
import urllib

app = Flask(__name__)
app.debug = True

@app.route('/')
@app.route('/index')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def render_prediction():
    file=request.files['image']
    img = misc.imread(file)

    num_colors = 16
    try:
        num_colors = int(request.form['myRange'])
    except Exception as e:
        print('Could not convert colors, using 16 as default')

    posterized_image = posterize_numpy(img, num_colors)

    posterized_image = posterized_image.getvalue().encode("base64").rstrip('\n')
    original_image = numpy_to_img(img).getvalue().encode("base64").rstrip('\n')
    return render_template('prediction_fancy.html',
                            orig_image=urllib.quote(original_image),
                            img_data=urllib.quote(posterized_image),
                            num_colors=num_colors)

if __name__ == '__main__':
    app.run()
