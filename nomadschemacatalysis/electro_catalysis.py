import numpy as np
import os

from ase.data import chemical_symbols

from nomad.metainfo import (
    Quantity,
    Section,
    SubSection,
    Package)

from nomad.datamodel.metainfo.eln import (
    #CompositeSystem,
    Measurement)

from nomad.datamodel.metainfo.basesections import CompositeSystem, System

from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.results import (Results, Properties, CatalyticProperties,
                                     CatalystCharacterization, CatalystSynthesis, Reactivity)

from nomad.datamodel.results import Product, Reactant

from nomad.datamodel.data import EntryData, UseCaseElnCategory

from .catalytic_measurement import (
    CatalyticReactionData, CatalyticReactionData_core, Feed, Reagent, Conversion, Rates, Reactor_setup,
    ECatalyticReactionData, PotentiostaticMeasurement )


from .schema import CatalystSample

m_package = Package(name='e-catalysis')

def add_catalyst(archive):
    '''Adds metainfo structure for catalysis data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = CatalyticProperties()
    if not archive.results.properties.catalytic.catalyst_characterization:
        archive.results.properties.catalytic.catalyst_characterization = CatalystCharacterization()
    if not archive.results.properties.catalytic.catalyst_synthesis:
        archive.results.properties.catalytic.catalyst_synthesis = CatalystSynthesis()


def add_activity(archive):
    '''Adds metainfo structure for catalysis activity test data.'''
    if not archive.results:
        archive.results = Results()
    if not archive.results.properties:
        archive.results.properties = Properties()
    if not archive.results.properties.catalytic:
        archive.results.properties.catalytic = CatalyticProperties()
    if not archive.results.properties.catalytic.reactivity:
        archive.results.properties.catalytic.reactivity = Reactivity(

class ElectrocatalystSample(CatalystSample, EntryData):

    m_def = Section(
        label='Electrocatalyst Sample',
        # a_eln=dict(  # lane_width='400px',
        #     hide=['cas_uri', 'cas_number', 'cas_name', 'inchi', 'inchi_key', 'smile',
        #           'canonical_smile', 'cas_synonyms', 'molecular mass']),
        categories=[UseCaseElnCategory])

    electrochemically_active_surface_area = Quantity(
        type=np.float64,
        unit=("m**2/g"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='m**2/g',
        ))

    electrode_area_geometric = Quantity(
        type=np.dtype(np.float64),
        unit=("cm**2"),
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='cm**2',
        ))

    catalyst_mass = Quantity(
        type=np.dtype(np.float64),
        unit=("mg"),
        description="""
        loading of catalyst on electrode support
        """,
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='mg',
        ))

    sample_support = Quantity(
        type=str,
        shape=[],
        description="""
          electrode support
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['carbon paper', 'unknown'])))

    sample_binder = Quantity(
        type=str,
        shape=[],
        description="""
          binder to fix catalyst to support
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(
                suggestions=['Nafion', 'unknown'])))

    def normalize(self, archive, logger):
        super(ElectrocatalystSample, self).normalize(archive, logger)


class ElectroCatalyticReaction(EntryData):

    m_def = Section(
        label='Electrocatalytic Measurement',
        categories=[UseCaseElnCategory],
    )

    sample_reference = Quantity(
        type=ElectrocatalystSample,
        description="""
        link to an electrocatalyst sample entry
        """,
        a_eln=dict(component='ReferenceEditQuantity')
    )

    reaction_class = Quantity(
        type=str,
        description="""
        highlevel classification of reaction
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(suggestions=[
            'Oxidation', 'Reduction']
        )))

    reaction_name = Quantity(
        type=str,
        description="""
          name of reaction
          """,
        a_eln=dict(
            component='EnumEditQuantity', props=dict(suggestions=[
                'CO2RR', 'HER', 'OER'])))

    existing_experiment_handbook = Quantity(
        description="""
        was the experiment performed according to a handbook
        """,
        type=str,
        shape=[],
        a_eln=dict(component='FileEditQuantity')
    )

    institute = Quantity(
        type=str,
        shape=[],
        description="""
        The institution at which the measurement was performed.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['Fritz-Haber-Institut Berlin / Abteilung AC',
                         'Fritz-Haber-Institut Berlin / ISC',
                         'TU Berlin, BASCat', 'HZB', 'CATLAB']))
    )

    experimenter = Quantity(
        type=str,
        shape=[],
        description="""
        person that performed or started the measurement
        """,
        a_eln=dict(component='EnumEditQuantity')
    )

    data_file = Quantity(
        type=str,
        description="""
        An excel or csv file that contains results of a electrocatalytic measurement with
        corrected potential, current, gas feed composition, partial current density,
        and Faradeic Efficiency (product selectivities)
        """,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    electrolyte = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['Water', 'Base', 'Acid'])))

    electrolyte_concentration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='mol/L',
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='mol/L'))

    electrolyte_gas_saturation = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['Argon', 'N2', 'CO2'])))

    stirring_speed = Quantity(
        type=np.float64,
        shape=[],
        unit='1/s',
        a_eln=dict(
            component='NumberEditQuantity', defaultDisplayUnit='1/s'))

    pH = Quantity(
        type=np.dtype(np.float64),
        unit=("pH"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='pH'))

    measurement_type = Quantity(
        type=str,
        shape=[],
        a_eln=dict(component='StringEditQuantity',
                   props=dict(suggestions=['constant potential', 'pulsed potential', 'constant current']))
    )

    potential = Quantity(
        type=np.float64,
        unit=("mV"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='mV'))

    measurement_duration = Quantity(
        type=np.float64,
        unit=("s"),
        a_eln=dict(component='NumberEditQuantity', defaultDisplayUnit='s'))

    electrode_support = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    electrocatalyst_binder = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    cell_type = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['H-type', 'flow', 'other'])))
    
    membrane = Quantity(
        type=str,
        a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['other'])))

    counter_electrode = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['Pt'])))
    reference_electrode = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
            suggestions=['RHE', 'Ag/AgCl'])))

    e_feed = SubSection(section_def=Feed)

    e_reaction_data = SubSection(section_def=ECatalyticReactionData)

    potentiostatic_measurement = SubSection(section_def=PotentiostaticMeasurement)
    measurement_details = SubSection(section_def=Measurement)

    def normalize(self, archive, logger):
        super(ElectroCatalyticReaction, self).normalize(archive, logger)

        if not self.data_file:
            return

        if os.path.splitext(
                self.data_file)[-1] != ".csv" and os.path.splitext(
                self.data_file)[-1] != ".xlsx":
            raise ValueError("Unsupported file format. Only xlsx and .csv files")

        if self.data_file.endswith(".csv"):
            with archive.m_context.raw_file(self.data_file) as f:
                import pandas as pd
                data = pd.read_csv(f.name).dropna(axis=1, how='all')
        elif self.data_file.endswith(".xlsx"):
            with archive.m_context.raw_file(self.data_file) as f:
                import pandas as pd
                data = pd.read_excel(f.name, sheet_name=0)

        data.dropna(axis=1, how='all', inplace=True)
        feed = Feed()
        e_cat_data = ECatalyticReactionData()
        reactants = []
        reactant_names = []
        products = []
        product_names = []
        faradaic_efficiencies = []
        partial_current_densities = []
        number_of_runs = 0
        for col in data.columns:

            if len(data[col]) < 1:
                continue
            col_split = col.split(" ")
            if len(col_split) < 2:
                continue

            if len(data[col]) > number_of_runs:
                number_of_runs = len(data[col])

            if col_split[0] == "x":
                reactant = Reactant(name=col_split[1],
                                    amount=data[col])
                reactant_names.append(col_split[1])
                reactants.append(reactant)

            if col_split[0] == "ToS":
                e_cat_data.time_on_stream = data[col]

            if col_split[0] == "gas_flow":
                feed.flow_rates = data[col]

            if col_split[0] == "partial current density":  # corresponds to reaction rate
                partial_current_densities = Ecat_Product(name=col_split[3],
                                                         partial_current_density=data[col])

                Ecat_Product.partial_current_density = data[col]
                partial_current_densities.append(partial_current_densities)

            if col_split[0] == "F.E.":  # Faradeic Efficiency (selectivity)
                product = Ecat_Product(name=col_split[1])
                for i, p in enumerate(products):
                    if p.name == col_split[1]:
                        product = products.pop(i)
                        break

                products.append(product)

                product.faradaic_efficiency = data[col]
                product_names.append(col_split[1])
                faradaic_efficiencies.append(data[col])

        for p in products:
            if p.selectivity is None or len(p.selectivity) == 0:
                p.selectivity = number_of_runs * [0]

        feed.reactants = reactants
        feed.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        e_cat_data.ecat_products = products
        e_cat_data.runs = np.linspace(0, number_of_runs - 1, number_of_runs)
        e_cat_data.product = faradaic_efficiencies

        self.feed = feed
        self.e_reaction_data = e_cat_data

        add_activity(archive)

        # if feed.reactants is not None:
        #     archive.results.properties.catalytic.reactivity.reactants = reactants
        #     archive.results.properties.catalytic.activity.conversion='cat_data.conversion/0:1/conversion_product_based'
        #     archive.results.properties.catalytic.reactivity.conversion_names = conversion_names
        # if products is not None:
        #     archive.results.properties.catalytic.reactivity.products = products
        if self.reaction_name:
            archive.results.properties.catalytic.reactivity.reaction_name = self.reaction_name
            archive.results.properties.catalytic.reactivity.reaction_class = self.reaction_class

m_package.__init_metainfo__()