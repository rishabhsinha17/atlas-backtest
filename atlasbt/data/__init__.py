"""atlasbt.data — point-in-time data layer.

Public entry points:
  * atlasbt.data.portal.Portal        -- implements core.protocols.DataPortal
  * atlasbt.data.store.ParquetStore   -- on-disk canonical store (cache/store)
  * atlasbt.data.qa                   -- data quality checks / reports
  * python -m atlasbt.data.fetch      -- downloader CLI

See docs/DATA.md for sources, fidelity and point-in-time conventions.
"""
