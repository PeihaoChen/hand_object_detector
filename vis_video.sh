img_dir="$1"

cd "$img_dir"
# input_img_contact="$img_dir/*_contact.png"
# input_img_det="$img_dir/*_det.png"



# cat $(ls -v *_contact.png) | ffmpeg -f image2pipe -framerate 10 -i - -c:v libx264 -vf "fps=30,format=yuv420p" contact_video.mp4
# cat $(ls -v *_det.png) |  ffmpeg -f image2pipe -framerate 10 -i - -c:v libx264 -vf "fps=30,format=yuv420p" det_video.mp4


# ffmpeg -i contact_video.mp4 -i det_video.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" merge_video.mp4


# For contact_video.mp4
cat $(ls -v *_contact.png) | ffmpeg -y -f image2pipe -framerate 10 -i - -s 1280x720 -b:v 1000k -c:v libx265 -crf 28 -vf "fps=30,format=yuv420p" contact_video_temp.mp4

# For det_video.mp4
cat $(ls -v *_det.png) | ffmpeg -y -f image2pipe -framerate 10 -i - -s 1280x720 -b:v 1000k -c:v libx265 -crf 28 -vf "fps=30,format=yuv420p" det_video_temp.mp4

# Combine the two videos side by side
ffmpeg -y -i contact_video_temp.mp4 -i det_video_temp.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" merge_video.mp4
