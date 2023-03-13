from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/models/<path:path>')
def get_file(path):
    return send_from_directory('models', path)

if __name__ == '__main__':
    print("Starting cache server...`")
    app.run(host='0.0.0.0')