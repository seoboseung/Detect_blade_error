# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app, supports_credentials=True)

videos = preload_data()
set_videos(videos)

inference_api = InferenceAPI()


@app.route("/healthy")
def healthy() -> Response:
    return make_response("OK", 200)


@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        raise ValueError("resource not found")


@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    data = request.json
    session_id = data["session_id"]
    start_frame_index = data.get("start_frame_index", 0)
    video_filename = data.get("video_filename", "unknown.mp4")  # ⬅️ 추가

    boundary = "frame"
    frame = gen_track_with_mask_stream(boundary, session_id, start_frame_index, video_filename)

    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(boundary, session_id, start_frame_index, video_filename):
    import json
    import os

    all_chunks = []
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    with inference_api.autocast_context():
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        for chunk in inference_api.propagate_in_video(request=request):
            chunk_json = chunk.to_json()
            all_chunks.append(json.loads(chunk_json))

            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk_json.encode("utf-8"),
            ).get_message()

    # 전체 저장 (video_filename 추가)
    output_data = {
        "video_filename": video_filename,
        "session_id": session_id,
        "frames": all_chunks,
    }

    output_path = os.path.join(output_dir, f"{session_id}_masks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved masks to {output_path}")

# # 기존 코드 유지
# @app.route("/propagate_in_video", methods=["POST"])
# def propagate_in_video() -> Response:
#     data = request.json
#     session_id = data["session_id"]
#     start_frame_index = data.get("start_frame_index", 0)

#     boundary = "frame"
#     frame = gen_track_with_mask_stream(boundary, session_id, start_frame_index)

#     return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


# def gen_track_with_mask_stream(boundary: str, session_id: str, start_frame_index: int):
#     import json
#     import os

#     all_chunks = []
#     output_dir = "outputs"
#     os.makedirs(output_dir, exist_ok=True)

#     with inference_api.autocast_context():
#         request = PropagateInVideoRequest(
#             type="propagate_in_video",
#             session_id=session_id,
#             start_frame_index=start_frame_index,
#         )

#         for chunk in inference_api.propagate_in_video(request=request):
#             chunk_json = chunk.to_json()

#             # 저장용 리스트에 추가
#             all_chunks.append(json.loads(chunk_json))

#             # 클라이언트로 stream 전송
#             yield MultipartResponseBuilder.build(
#                 boundary=boundary,
#                 headers={
#                     "Content-Type": "application/json; charset=utf-8",
#                     "Frame-Current": "-1",
#                     "Frame-Total": "-1",
#                     "Mask-Type": "RLE[]",
#                 },
#                 body=chunk_json.encode("utf-8"),
#             ).get_message()

#     # 모든 프레임 마스크 저장
#     output_path = os.path.join(output_dir, f"{session_id}_masks.json")
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(all_chunks, f, ensure_ascii=False, indent=2)

#     print(f"[INFO] Saved masks to {output_path}")


# # TOOD: Protect route with ToS permission check
# @app.route("/propagate_in_video", methods=["POST"])
# def propagate_in_video() -> Response:
#     data = request.json
#     args = {
#         "session_id": data["session_id"],
#         "start_frame_index": data.get("start_frame_index", 0),
#     }

#     boundary = "frame"
#     frame = gen_track_with_mask_stream(boundary, **args)
#     return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


# def gen_track_with_mask_stream(
#     boundary: str,
#     session_id: str,
#     start_frame_index: int,
# ) -> Generator[bytes, None, None]:
#     with inference_api.autocast_context():
#         request = PropagateInVideoRequest(
#             type="propagate_in_video",
#             session_id=session_id,
#             start_frame_index=start_frame_index,
#         )

#         for chunk in inference_api.propagate_in_video(request=request):
#             yield MultipartResponseBuilder.build(
#                 boundary=boundary,
#                 headers={
#                     "Content-Type": "application/json; charset=utf-8",
#                     "Frame-Current": "-1",
#                     # Total frames minus the reference frame
#                     "Frame-Total": "-1",
#                     "Mask-Type": "RLE[]",
#                 },
#                 body=chunk.to_json().encode("UTF-8"),
#             ).get_message()


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
