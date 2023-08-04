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

from nomad.metainfo import (Quantity, SubSection, Section)
from nomad.datamodel.data import ArchiveSection


class Reagent(ArchiveSection):
    m_def = Section(label_quantity='name', description='a chemical substance present in the initial reaction mixture')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'), description="reagent name")
    gas_fraction = Quantity(
        type=np.dtype(np.float64), shape=['*'],
        description='volumetric fraction of reactant in feed, in %')


class Conversion(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))
    reference = Quantity(type=Reagent, a_eln=dict(component='ReferenceEditQuantity'))
    type = Quantity(type=str, a_eln=dict(component='StringEditQuantity', props=dict(
        suggestions=['product_based', 'reactant_based', 'unknown'])))
    conversion = Quantity(type=np.dtype(np.float64), shape=['*'])
    conversion_product_based = Quantity(type=np.dtype(np.float64), shape=['*'])
    conversion_reactant_based = Quantity(type=np.dtype(np.float64), shape=['*'])


class Reactant(Reagent):
    m_def = Section(label_quantity='name', description='a reagent that has a conversion in a reaction that is not null')

    conversion = SubSection(section_def=Conversion)


class Feed(ArchiveSection):
    m_def = Section(a_plot=[
        {
            "title": "Feed",
            "label": "gas composition", 'x': 'runs', 'y': ['reagents/:/gas_fraction'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}}])

    space_velocity = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='1/hour')

    set_pressure = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='bar')

    flow_rates = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='mL/minutes')

    catalyst_mass = Quantity(
        type=np.dtype(np.float64), shape=[], unit='g')

    runs = Quantity(type=np.dtype(np.float64), shape=['*'])

    sampling_frequency = Quantity(type=np.dtype(np.float64), shape=[],
                                  description='duration of a single run')

    reagents = SubSection(section_def=Reagent, repeats=True)


class Rates(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, a_eln=dict(component='StringEditQuantity'))

    reaction_rate = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        unit='mmol/g/hour',
        description='reaction rate for mmol of product (or reactant) formed (depleted) per catalyst (g) per time (hour)')
    specific_mass_rate = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='mmol/g/hour',
        description='reaction rate normalized by active (metal) catalyst mass (in gram), instead of mass of total catalyst')
    specific_sa_rate = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='mmol/m**2/hour',
        description='reaction rate normalized by active (metal) surface area of catalyst, instead of mass of total catalyst')
    space_time_yield = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        unit='g/g/hour',
        description='product amount formed (in g), per total catalyst (g) per time (hour)')
    rate = Quantity(
        type=np.dtype(np.float64), shape=['*'], unit='g/g/hour',
        description='reactant amount converted (in g), per total catalyst (g) per time (hour)'
    )
    # turn_over_number = Quantity(type=np.dtype(np.float64), shape=['*'])
    turn_over_frequency = Quantity(type=np.dtype(np.float64), shape=['*'], unit='1/hour')


class Product(Rates, ArchiveSection):
    m_def = Section(label_quantity='name')

    selectivity = Quantity(type=np.dtype(np.float64), shape=['*'])
    product_yield = Quantity(type=np.dtype(np.float64), shape=['*'])


class Reactor_setup(ArchiveSection):
    m_def = Section(label_quantity='name')
    name = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'))
    reactor_volume = Quantity(type=np.dtype(np.float64), shape=[], unit='ml',
                              a_eln=dict(component='NumberEditQuantity'))
    bed_length = Quantity(type=np.dtype(np.float64), shape=[], unit='mm',
                          a_eln=dict(component='NumberEditQuantity'))
    reactor_cross_section_area = Quantity(type=np.dtype(np.float64), shape=[], unit='mm**2',
                                          a_eln=dict(component='NumberEditQuantity'))
    reactor_shape = Quantity(type=str, shape=[], a_eln=dict(component='EnumEditQuantity'),
                             props=dict(suggestions=['cylindric', 'spherical']))
    diluent = Quantity(
        type=str,
        shape=[],
        description="""
        component that is mixed with the catalyst to dilute and prevent transport
        limitations and hot spot formation
        """,
        a_eln=dict(component='EnumEditQuantity', props=dict(
            suggestions=['SiC', 'SiO2', 'unknown']))
    )


class CatalyticReactionData(ArchiveSection):
    m_def = Section(a_plot=[
        {
            "label": "Selectivity [%]",
            'x': 'runs',
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Conversion X [%]",
            'x': 'runs',
            'y': ['reactants_conversions/:/conversion_product_based'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Conversion x_p ",
            'x': 'runs',
            'y': ['reactants_conversions/:/conversion_reactant_based'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Carbon Balance",
            'x': 'runs',
            'y': ['c_balance'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Temperature",
            'x': 'runs',
            'y': ['temperature'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "S_X plot",
            # "mode": "markers",
            'x': ['reactants_conversions/0:1/conversion'],
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True},
            "lines": [
                {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"}]
        },
        {
            "label": "S_X plot 2",
            'x': ['reactants_conversions/1:2/conversion'],
            'y': ['products/:/selectivity'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True},
            "lines": [{"mode": "markers"}, {"mode": "markers"}, {"mode": "markers"},
                      {"mode": "markers"}]
        },
        {
            "label": "rate",
            'x': 'runs',
            'y': ['rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "Temp vs. rate",
            'x': 'temperature',
            'y': ['rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}},
        {
            "label": "TOS vs. rate",
            'x': ['time_on_stream', 'runs'],
            'y': ['rates/:/reaction_rate','rates/:/reaction_rate'],
            'layout': {"showlegend": True,
                       'yaxis': {
                           "fixedrange": False}, 'xaxis': {
                           "fixedrange": False}}, "config": {
                "editable": True, "scrollZoom": True}
        }
    ]
    )

    temperature = Quantity(
        type=np.dtype(
            np.float64), shape=['*'], unit='°C')

    pressure = Quantity(
        type=np.dtype(np.float64),
        shape=['*'],
        unit='bar'
    )

    c_balance = Quantity(
        type=np.dtype(
            np.float64), shape=['*'])

    runs = Quantity(type=np.dtype(np.float64), shape=['*'])
    time_on_stream = Quantity(type=np.dtype(np.float64), shape=['*'], unit='hr')

    products = SubSection(section_def=Product, repeats=True)

    reactants_conversions = SubSection(section_def=Conversion, repeats=True)
    rates = SubSection(section_def=Rates, repeats=True)
