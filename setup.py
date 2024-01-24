from setuptools import setup


setup(
    use_scm_version={
        "write_to": "particle_tracking_manager/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    entry_points={"console_scripts": ["ptm=particle_tracking_manager.cli:main"]},
)
