let stream;
    let recorder;
    let videoPreview = document.getElementById('video-preview');
    let startRecordingButton = document.getElementById('start-recording');
    let stopRecordingButton = document.getElementById('stop-recording');
    let downloadLink = document.getElementById('download-link');


    startRecordingButton.addEventListener('click', async () => {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoPreview.srcObject = stream;
        recorder = new MediaRecorder(stream);
        let chunks = [];

        recorder.ondataavailable = (e) => {
            chunks.push(e.data);
        };

        recorder.onstop = () => {
            let blob = new Blob(chunks, { type: 'video/webm' });
            let url = URL.createObjectURL(blob);
            
            let formData = new FormData();
            formData.append('file', blob);
            token = "L8XFVDUUFQYHIZTZCEW8RI1UJ3OF523E"

            header = {
                'Content-Type': 'multipart/form-data'
            }
            fetch('http://localhost:8001/engine/video?token=L8XFVDUUFQYHIZTZCEW8RI1UJ3OF523E', {
                method: 'POST',
                body: formData
            })
            .then(res=>res.json()).then(data => {
                console.log(data.data);
                alert(data.data)
            })
        };

        recorder.start();
        startRecordingButton.disabled = true;
        stopRecordingButton.disabled = false;
    });

    stopRecordingButton.addEventListener('click', () => {
        recorder.stop();
        stream.getTracks().forEach(track => track.stop());
        videoPreview.srcObject = null;
        startRecordingButton.disabled = false;
        stopRecordingButton.disabled = true;
    });