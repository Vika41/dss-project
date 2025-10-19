import json

def export_predictions(predictions_by_pid, output_path):
    output = []
    for entry in predictions_by_pid:
        pid = entry["pid"]
        tracks = entry["tracks"]
        output.append({
            "pid": pid,
            "tracks": tracks
        })
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)