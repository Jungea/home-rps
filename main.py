from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)


@app.route('/page/rps')
def rps_page():
    return render_template('rps.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
