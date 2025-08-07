class NoSuitablePlansError(Exception):
    def __init__(self, constraints, max_chunk_duration, tmin, tmax):
        msg = f"No chunks of duration {max_chunk_duration} satisfy the constraints {constraints} between {tmin} and {tmax}"
        super().__init__(msg)
