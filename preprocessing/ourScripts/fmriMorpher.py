import mne

#add your local path into the file checkPath.py
from checkPath import checkPath
checkPath()

from projects.facerecognition_dtu import utils
from projects.facerecognition_dtu.config import Config

for subject_id in range(1,3):

    io = utils.SubjectIO(subject_id)

    src = mne.read_source_spaces(io.data.get_filename(stage="forward", forward="mne", suffix="src"))

    fmri_morph = mne.compute_source_morph(src, subjects_dir=Config.path.FREESURFER)
    #create a folder if you don't have one
    fmri_morph.save(f"data/fmriMorphers/sub-{subject_id:02d}")
