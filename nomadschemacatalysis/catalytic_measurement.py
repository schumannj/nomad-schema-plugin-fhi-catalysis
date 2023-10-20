#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

from nomad.metainfo import (Quantity, SubSection, Section, MSection)
from nomad.datamodel.data import ArchiveSection

from nomad.datamodel.metainfo.plot import PlotSection, PlotlyFigure
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json

class Reagent(ArchiveSection):
    m_def = Section(label_quantity='name', description='a chemical substance present in the initial reaction mixture')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'), description="reagent name")
    gas_concentration_in = Quantity(
        type=np.float64, shape=['*'],
        description='Volumetric fraction of reactant in feed.')
    flow_rate = Quantity(
        type=np.float64, shape=['*'], unit='mL/minutes',
        description='Flow rate of reactant in feed.')


class Conversion(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    reference = Quantity(type=Reagent, a_eln=dict(component='ReferenceEditQuantity'))
    type = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
        suggestions=['product_based', 'reactant_based', 'unknown'])))
    conversion = Quantity(type=np.float64, shape=['*'])
    conversion_product_based = Quantity(type=np.float64, shape=['*'])
    conversion_reactant_based = Quantity(type=np.float64, shape=['*'])


class Reactant(Reagent):
    m_def = Section(label_quantity='name', description='a reagent that has a conversion in a reaction that is not null')

    conversion = SubSection(section_def=Conversion)


class Feed(PlotSection, ArchiveSection):  
    # m_def = Section(a_plot=[
    #     {
    #         "title": "Feed",
    #         "label": "gas composition", 'x': 'runs', 'y': ['reagents/:/gas_concentration_in'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}}]
    #             )

    space_velocity = Quantity(
        type=np.float64, shape=['*'], unit='1/hour')  #, a_eln=dict(component='NumberEditQuantity'))

    set_pressure = Quantity(
        type=np.float64, shape=['*'], unit='bar')  #, a_eln=dict(component='NumberEditQuantity'))
    
    set_temperature = Quantity(
        type=np.float64, shape=['*'], unit='K')  #, a_eln=dict(component='NumberEditQuantity'))

    flow_rates_total = Quantity(
        type=np.float64, shape=['*'], unit='mL/minutes')  #, a_eln=dict(component='NumberEditQuantity'))

    catalyst_mass = Quantity(
        type=np.float64, shape=[], unit='g', a_eln=dict(component='NumberEditQuantity'))

    runs = Quantity(type=np.float64, shape=['*'])

    sampling_frequency = Quantity(
        type=np.float64, shape=[], unit='Hz',
        description='The number of runs per time.',
        a_eln=dict(component='NumberEditQuantity'))

    reagents = SubSection(section_def=Reagent, repeats=True)

    def normalize(self, archive, logger):
        super(Feed, self).normalize(archive, logger)

        
        #if self.set_temperature and self.runs is not None:
        fig = px.scatter(x=self.runs, y=self.set_temperature)
        #fig.update_layout(title_text="Temperature", xaxis="measurement points", yaxis="Temperature (K)")
        self.figures.append(PlotlyFigure(label='Temperature', figure=fig.to_plotly_json()))

        # if self.reagents[0].gas_concentration_in and self.runs is not None:
        #     fig = px.scatter(x=self.runs, y=self.reagents[0].gas_concentration_in)
        #     for i,n in enumerate(self.reagents):
        #         if i > 0:
        #             fig.add_trace=(self.reagents[i].gas_concentration_in)
        #     self.figures.append(PlotlyFigure(label='gas concentration', figure=fig.to_plotly_json()))
        # elif self.reagents[0].flow_rate and self.runs is not None:
        #     fig = px.scatter(x=self.runs, y=self.reagents[0].flow_rate)
        #     for i,n in enumerate(self.reagents):
        #         if i > 0:
        #             fig.add_trace=(self.reagents[i].flow_rate)
        #     self.figures.append(PlotlyFigure(label='gas flow rate', figure=fig.to_plotly_json()))

# class Pretreatment(Feed):
#     m_def = Section(
#     a_plot=[{
#         "title": "Temperature",
#         "label": "Temperature", 'x': 'runs', 'y': ['set_temperature'],
#         'layout': {"showlegend": True,
#                    'yaxis': {
#                            "fixedrange": False}, 'xaxis': {
#                            "fixedrange": False}}, "config": {
#                 "editable": True, "scrollZoom": True}},
#         {
#         "title": "Pretreatment Feed",
#         "label": "gas composition", 'x': 'runs', 'y': ['reagents/:/gas_concentration_in'],
#         'layout': {"showlegend": True,
#                    'yaxis': {
#                            "fixedrange": False}, 'xaxis': {
#                            "fixedrange": False}}, "config": {
#                 "editable": True, "scrollZoom": True}},
#                 ]
#     )
    

class Rates(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    reaction_rate = Quantity(
        type=np.float64,
        shape=['*'],
        unit='mmol/g/hour',
        description='The reaction rate for mmol of product (or reactant) formed (depleted) per catalyst (g) per time (hour).')
    specific_mass_rate = Quantity(
        type=np.float64, shape=['*'], unit='mmol/g/hour',
        description='The specific reaction rate normalized by active (metal) catalyst mass, instead of mass of total catalyst.')
    specific_surface_area_rate = Quantity(
        type=np.float64, shape=['*'], unit='mmol/m**2/hour',
        description='The specific reaction rate normalized by active (metal) surface area of catalyst, instead of mass of total catalyst.')
    space_time_yield = Quantity(
        type=np.float64,
        shape=['*'],
        unit='g/g/hour',
        description='The amount of product formed (in g), per total catalyst (g) per time (hour).')
    rate = Quantity(
        type=np.float64, shape=['*'], unit='g/g/hour',
        description='The amount of reactant converted (in g), per total catalyst (g) per time (hour).'
    )
    turn_over_frequency = Quantity(
        type=np.float64, shape=['*'], unit='1/hour',
        description='The turn oder frequency, calculated from mol of reactant or product, per number of sites, over time.')


class Product(Rates, ArchiveSection):
    m_def = Section(label_quantity='name')

    selectivity = Quantity(type=np.float64, shape=['*'])
    product_yield = Quantity(type=np.float64, shape=['*'])


class Reactor_setup(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'))
    reactor_volume = Quantity(type=np.float64, shape=[], unit='ml',
                              a_eln=dict(component='NumberEditQuantity'))
    bed_length = Quantity(type=np.float64, shape=[], unit='mm',
                          a_eln=dict(component='NumberEditQuantity'))
    reactor_cross_section_area = Quantity(type=np.float64, shape=[], unit='mm**2',
                                          a_eln=dict(component='NumberEditQuantity'))
    reactor_diameter = Quantity(type=np.float64, shape=[], unit='mm',
                                          a_eln=dict(component='NumberEditQuantity'))
    reactor_shape = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'),
                             props=dict(suggestions=['cylindric', 'spherical']))
    diluent = Quantity(
        type=str,
        shape=[],
        description="""
        A component that is mixed with the catalyst to dilute and prevent transport
        limitations and hot spot formation.
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['SiC', 'SiO2', 'unknown']))
    )
    diluent_sievefraction_high = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    diluent_sievefraction_low = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_mass = Quantity(
        type=np.float64, shape=[], unit='g',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_sievefraction_high = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    catalyst_sievefraction_low = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))
    particle_size = Quantity(
        type=np.float64, shape=[], unit='micrometer',
        a_eln=dict(component='NumberEditQuantity'))

class CatalyticReactionData_core(ArchiveSection):
    temperature = Quantity(
        type=np.float64, shape=['*'], unit='°C')

    pressure = Quantity(
        type=np.float64, shape=['*'], unit='bar')

    runs = Quantity(type=np.float64, shape=['*'])
    time_on_stream = Quantity(type=np.float64, shape=['*'], unit='hour')

    reactants_conversions = SubSection(section_def=Conversion, repeats=True)
    rates = SubSection(section_def=Rates, repeats=True)

class CatalyticReactionData(CatalyticReactionData_core, ArchiveSection):
    m_def = Section(
    #     a_plot=[
    #     {
    #         "label": "Selectivity [%]",
    #         'x': 'runs',
    #         'y': ['products/:/selectivity'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "Conversion X [%]",
    #         'x': 'runs',
    #         'y': ['reactants_conversions/:/conversion_product_based'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "Conversion x_p ",
    #         'x': 'runs',
    #         'y': ['reactants_conversions/:/conversion_reactant_based'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "Carbon Balance",
    #         'x': 'runs',
    #         'y': ['c_balance'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "Temperature",
    #         'x': 'runs',
    #         'y': ['temperature'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "S_X plot",
    #         # "mode": "markers",
    #         'x': ['reactants_conversions/0:1/conversion'],
    #         'y': ['products/:/selectivity'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True},
    #         "lines": [
    #             {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}]
    #     },
    #     {
    #         "label": "S_X plot 2",
    #         'x': ['reactants_conversions/1:2/conversion'],
    #         'y': ['products/:/selectivity'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True},
    #         "lines": [{"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"},
    #                   {"mode": "markers"}]
    #     },
    #     {
    #         "label": "rate",
    #         'x': 'runs',
    #         'y': ['rates/:/reaction_rate'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "Temp vs. rate",
    #         'x': 'temperature',
    #         'y': ['rates/:/reaction_rate'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}},
    #     {
    #         "label": "TOS vs. rate",
    #         'x': ['time_on_stream', 'runs'],
    #         'y': ['rates/:/reaction_rate', 'rates/:/reaction_rate'],
    #         'layout': {"showlegend": True,
    #                    'yaxis': {
    #                        "fixedrange": False}, 'xaxis': {
    #                        "fixedrange": False}}, "config": {
    #             "editable": True, "scrollZoom": True}
    #     }
    # ]
    )

    c_balance = Quantity(
        type=np.dtype(
            np.float64), shape=['*'])
    products = SubSection(section_def=Product, repeats=True)


class Ecat_Product(MSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    partial_current_density = Quantity(type=np.dtype(np.float64), shape=['*'], unit='mA/cm**2')
    faradaic_efficiency = Quantity(type=np.dtype(np.float64), shape=['*'])


class PotentiostaticMeasurement(MSection):
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

class ECatalyticReactionData(MSection):

    m_def = Section(a_plot=[
        {
            "label": "Faradeic Efficiencies",
            'x': 'time',
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}])

    runs = Quantity(type=np.dtype(np.float64), shape=['*'])
    time_on_stream = Quantity(type=np.dtype(np.float64), shape=['*'], unit='h')

    set_potential = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='V')

    corrected_potential = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='V')

    resistivity = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='V A')

    current = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='mA')

    ecat_products = SubSection(section_def=Ecat_Product, repeats=True)
