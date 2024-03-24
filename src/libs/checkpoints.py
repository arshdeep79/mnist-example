import torch
import neptune 
import shutil

def saveCheckpoint(run, checkpoint):
    checkpointFile = f"./checkpoints/epoch-{checkpoint['epoch']}.pth"
    torch.save(checkpoint, checkpointFile)
    run[f"checkpoints/{checkpoint['epoch']}"].upload(checkpointFile)
    
def cleanup():
    shutil.rmtree('./checkpoints')

def loadLastCheckpoint():
    with neptune.init_project() as project:
        runs_table_df = project.fetch_runs_table(
            state="inactive"
        ).to_pandas()
    if not len(runs_table_df):
        return None
    
    succesfullRuns = runs_table_df[runs_table_df["sys/failed"] == False]

    if not succesfullRuns.size:
        return None
    
    run_id = succesfullRuns['sys/id'].values[0]

    
    run = neptune.init_run(
        with_id=run_id, 
        mode="read-only"
    )
    structure = run.get_structure()

    if 'checkpoints' not in structure: 
        return None 
    epochs = structure['checkpoints']

    epochsList = list(epochs.keys())
    epochsList.sort()
    lastEpoch = epochsList[-1]
  
    fileName = f"./checkpoints/epoch-{lastEpoch}.pth"
    run["checkpoints"][lastEpoch].download(destination=fileName)
    run.wait()
    run.stop()
    return torch.load(fileName) 