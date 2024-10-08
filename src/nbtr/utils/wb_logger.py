import wandb

class WandBLogger:
    def __init__(self, enabled, project, id, name, config=None):
        self.enabled = enabled
        self.project = project
        self.id = id
        self.name = name
        self.config = config if config else {}
        self.run = None

    def __enter__(self):
        if self.enabled:
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                id=self.id,
                resume="allow",
                config=self.config
            )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.run:
            self.run.finish()
            
        if exc_type:
            raise
        
        return True
    
    def log(self, message):
        if self.run:
            self.run.log(message)