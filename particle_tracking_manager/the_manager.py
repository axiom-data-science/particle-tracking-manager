import datetime
import json
import logging
import pathlib
from abc import abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Union

import appdirs
import pandas as pd
import xarray as xr
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Extra,
    Field,
    AfterValidator,
    create_model,
    field_validator,
    model_validator,
    validator,
)
from typing_extensions import Self

from .config import AdditionalValidationModel, ParticleTrackingState, PTMConfig


class ParticleTrackingManager:
    """Manager class that controls particle tracking model."""
    
    # TODO: update docs and tests to not demonstrate doing things in steps
    # since won't be able to anymore

    def __init__(self, **kwargs):
        # this inputs everything from PTMConfig to *.config.*
        self.config = PTMConfig(**kwargs)
        self.logger = logging.getLogger(__name__)
        self._setup_logger()
        self._setup_interpolator()
        self.state = ParticleTrackingState()

        # Run additional validation
        AdditionalValidationModel(config=self.config, logger=self.logger)


    def _setup_logger(self):
        """Setup logger."""
        logfile_name = self.config.output_file + ".log"
        self.file_handler = logging.FileHandler(logfile_name)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.info(f"Output filename: {self.config.output_file}")
        self.logger.info(f"Log filename: {logfile_name}")
        self.logfile_name = logfile_name

    def _setup_interpolator(self):
        """Setup interpolator."""
        if self.config.use_cache:
            cache_dir = pathlib.Path(appdirs.user_cache_dir(appname="particle-tracking-manager", appauthor="axiom-data-science"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            if self.config.interpolator_filename is None:
                self.config.interpolator_filename = cache_dir / pathlib.Path(f"{self.config.ocean_model}_interpolator").with_suffix(".pickle")
            else:
                self.config.interpolator_filename = pathlib.Path(self.config.interpolator_filename).with_suffix(".pickle")
            self.save_interpolator = True
            
            # change interpolator_filename to string
            self.config.interpolator_filename = str(self.config.interpolator_filename)
            
            if pathlib.Path(self.config.interpolator_filename).exists():
                self.logger.info(f"Loading the interpolator from {self.config.interpolator_filename}.")
            else:
                self.logger.info(f"A new interpolator will be saved to {self.config.interpolator_filename}.")
        else:
            self.save_interpolator = False
            self.logger.info("Interpolators will not be saved.")
    

    @abstractmethod
    def add_reader(self, **kwargs):
        """Here is where the model output is opened.
        
        Subclasses must implement this method and:
        
        * Set `self.has_added_reader = True` at the end of the method.
        """
        pass

    # TODO: change methods to _methods as appropriate

    @abstractmethod
    def seed(self, lon=None, lat=None, z=None):
        """Initialize the drifters in space and time.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_added_reader
        * Set `self.has_run_seeding = True` at the end of the method.
        """
        pass

    @abstractmethod
    def run(self):
        """Call model run function.
        
        Subclasses must implement this method and:
        
        * Raise a ValueError if not self.has_run_seeding
        * Set `self.has_run = True` at the end of the method.
        * Should close the file handler and logger.
        """
        pass

    def run_all(self):
        """Run all steps."""
        if not self.state.has_added_reader:
            self.add_reader()
        if not self.state.has_run_seeding:
            self.seed()
        if not self.state.has_run:
            self.run()

    def output(self):
        """Hold for future output function."""
        pass

    @abstractmethod
    def _config(self):
        """Model should have its own version which returns variable config."""
        pass

    # @abstractmethod
    # def _add_ptm_config(self):
    #     """Have this in the model class to modify config."""
    #     pass

    # @abstractmethod
    # def _add_model_config(self):
    #     """Have this in the model class to modify config."""
    #     pass

    # def _update_config(self) -> None:
    #     """Update configuration between model, PTM additions, and model additions."""
    #     self._add_ptm_config()
    #     self._add_model_config()

    @abstractmethod
    def show_config_model(self):
        """Define in child class."""
        pass

    def show_config(self, **kwargs) -> dict:
        """Show parameter configuration across both model and PTM."""
        # self._update_config()
        config = self.show_config_model(**kwargs)
        return config

    @abstractmethod
    def reader_metadata(self, key):
        """Define in child class."""
        pass

    @abstractmethod
    def query_reader(self):
        """Define in child class."""
        pass

    @abstractmethod
    def all_export_variables(self):
        """Output list of all possible export variables."""
        pass

    @abstractmethod
    def export_variables(self):
        """Output list of all actual export variables."""
        pass

    @property
    @abstractmethod
    def outfile_name(self):
        """Output file name."""
        pass