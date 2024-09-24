"""
torchrun --nproc-per-node=8 tools/caption/api_oci.py
"""
import os
import argparse
from flask import Flask, request, jsonify
import traceback

from video_caption_oci import predict_v2 as predict

app = Flask(__name__)


@app.route('/video_qa', methods=['POST'])
def video_qa():
    if 'video' not in request.files:
        return jsonify({'error': 'no video file found'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'no chosen file'}), 400

    if 'question' not in request.form:
        question = ""
    else:
        question = request.form['question']

    if question is None or question == "" or question == "@Caption":
        question = "Please describe the video in detail."

    print("Get question:", question)

    # if 'temperature' not in request.form:
    #     temperature = 0.001
    #     print("No temperature found, use default value 0.001")
    # else:
    #     temperature = float(request.form['temperature'])
    #     print("Get temperature:", temperature)
    
    temperature = float(request.form.get('temperature', 0.001))
    max_new_tokens = int(request.form.get('max_new_tokens', 2048))
    top_k = int(request.form.get('top_k', 1))
    do_sample = bool(request.form.get('do_sample', False))
    top_p = float(request.form.get('top_p', 0.1))

    print("Get temperature:", temperature)
    print("Get max_new_tokens:", max_new_tokens)
    print("Get top_k:", top_k)
    print("Get do_sample:", do_sample)
    print("Get top_p:", top_p)

    try:
        answer = predict(prompt=question, video_data=video.read(), temperature=temperature, max_new_tokens=max_new_tokens, top_k=top_k, do_sample=do_sample, top_p=top_p)
        return jsonify(
            {"answer": answer})
    except:
        traceback.print_exc()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    app.run(debug=False, host="0.0.0.0", port=5000+local_rank)
