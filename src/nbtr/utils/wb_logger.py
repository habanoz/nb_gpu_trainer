

class Backend:
    def log(self, message):
        raise NotImplementedError()
    
    def finish(self):
        raise NotImplementedError()
    
class CsvLoggerBackend(Backend):
    def __init__(self, file_name):
        super().__init__()
        self.csv_logger  = None
        self.file = open(file_name, "a")
    
    def log(self, message):
        
        if self.csv_logger is None:
            self.csv_logger = csv.DictWriter(self.file, fieldnames=list(message))
            self.csv_logger.writeheader()

        self.csv_logger.writerow(message)
    
    def finish(self):
        self.file.close()

class TrainingLogger:
    def __init__(self, log_to_backends, config):
        self.log_to_backends = log_to_backends
        self.loggers = []
        self.config = config

    def _config_logger(self, log_backend):
        if log_backend=='wandb':
            import wandb
            return wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                id=self.config.wandb_run_id,
                resume="allow",
                config=None
            )
        elif log_backend=='csv':
            return CsvLoggerBackend(f"{self.config.out_dir}/report.csv")
            
    def __enter__(self):
        for backend in self.log_to_backends:
            logger=self._config_logger(backend)
            self.loggers.append(logger)

        return self

    def __exit__(self, exc_type, exc_value, tb):
        for logger in self.loggers:
            logger.finish()
            
        if exc_type:
            raise
        
        return True
    
    def log(self, message):
        for logger in self.loggers:
            logger.log(message)