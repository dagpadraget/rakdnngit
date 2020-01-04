import trainer.task as task

# directory for temporary data
tmpdirname='c:\\tmp\\rakdnngit'

xbatch_size=32
xlearning_rate=0.01
xnum_epochs=1

class Argar:
    pcsim = True
    pcsimdir = tmpdirname
    batch_size=xbatch_size
    learning_rate=xlearning_rate
    num_epochs=xnum_epochs
    job_dir=tmpdirname+"\\jobdir"

argen=Argar()

task.train_and_evaluate(argen)
print("*************** training main - done")

task.test_model(argen)
print("*************** test main - done")
