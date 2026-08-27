nats-server -a 0.0.0.0 -p 4222             or                  nats-server -c nats-server.conf

python3 person_detector_json.py --model yolov8n_1024_fp16.nb --library ./libnn_yolov8n_1024_fp16.so --inputs rtsp://192.168.1.38:8554/stream1 --no-show
