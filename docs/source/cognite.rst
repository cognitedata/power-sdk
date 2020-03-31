Quickstart
==========

Create a Power Client object with:

    >>> from cognite.power import PowerClient
    >>> c = CogniteClient(api_key="<your-api-key>", project="<your project>",)

Read more about the `PowerClient`_ and the functionality it exposes below.


SDK Methods
===========

PowerClient
-----------
.. autoclass:: cognite.power.PowerClient
    :members:
    :member-order: bysource


Generic Power API
-----------------
These methods typically return classes which correspond to their specific API,
for example, `client.power_transformers.list()` will return a `PowerAssetList`_ with
`PowerTransformer`_.

Retrieve an asset by name
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: cognite.power._api.generic_power.GenericPowerAPI.retrieve_name

List assets
^^^^^^^^^^^
.. automethod:: cognite.power._api.generic_power.GenericPowerAPI.list

Search for assets
^^^^^^^^^^^^^^^^^
.. automethod:: cognite.power._api.generic_power.GenericPowerAPI.search


Base Data classes
-----------------

PowerAsset
^^^^^^^^^^

.. autoclass:: cognite.power.data_classes.PowerAsset
    :members:
    :show-inheritance:

PowerAssetList
^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.PowerAssetList
    :members:
    :show-inheritance:

Specific Power Asset Data classes
---------------------------------

ACLineSegment
^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.ACLineSegment
    :members:
    :show-inheritance:

Substation
^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.Substation
    :members:
    :show-inheritance:

PowerTransformer
^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.PowerTransformer
    :members:
    :show-inheritance:

PowerTransformerEnd
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.PowerTransformerEnd
    :members:
    :show-inheritance:

SynchronousMachine
^^^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.SynchronousMachine
    :members:
    :show-inheritance:

HydroGeneratingUnit
^^^^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.HydroGeneratingUnit
    :members:
    :show-inheritance:

WindGeneratingUnit
^^^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.data_classes.WindGeneratingUnit
    :members:
    :show-inheritance:

Terminal
^^^^^^^^
.. autoclass:: cognite.power.data_classes.Terminal
    :members:
    :show-inheritance:

Analog
^^^^^^
.. autoclass:: cognite.power.data_classes.Analog
    :members:
    :show-inheritance:

Power Corridor
--------------

PowerCorridor
^^^^^^^^^^^^^
.. autoclass:: cognite.power.power_corridor.PowerCorridor
    :members:
    :show-inheritance:


PowerCorridorComponent
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: cognite.power.power_corridor.PowerCorridorComponent
    :members:
    :show-inheritance:

Exceptions
^^^^^^^^^^
.. automodule:: cognite.power.exceptions
    :members:
    :show-inheritance:

Power Graph
-----------
Initialize a PowerGraph with all assets using `client.power_graph()`.

PowerGraph
^^^^^^^^^^
.. autoclass:: cognite.power.power_graph.PowerGraph
    :members:
    :show-inheritance:
