"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from flex.pool.pool import FlexPool

from flex.pool.primitives_tf import init_server_model_tf
from flex.pool.primitives_tf import deploy_server_model_tf
from flex.pool.primitives_tf import collect_clients_weights_tf
from flex.pool.primitives_tf import train_tf
from flex.pool.primitives_tf import set_aggregated_weights_tf
from flex.pool.primitives_tf import evaluate_model_tf

from flex.pool.primitives_pt import deploy_server_model_pt
from flex.pool.primitives_pt import collect_clients_weights_pt
from flex.pool.primitives_pt import set_aggregated_weights_pt
from flex.pool.primitives_pt import set_aggregated_diff_weights_pt
from flex.pool.primitives_pt import collect_client_diff_weights_pt

from flex.pool.aggregators import fed_avg
from flex.pool.aggregators import fed_avg_f
from flex.pool.aggregators import weighted_fed_avg
from flex.pool.aggregators import weighted_fed_avg_f
from flex.pool.aggregators import set_tensorly_backend

from flex.pool.decorators import init_server_model
from flex.pool.decorators import deploy_server_model
from flex.pool.decorators import collect_clients_weights
from flex.pool.decorators import aggregate_weights
from flex.pool.decorators import set_aggregated_weights
from flex.pool.decorators import evaluate_server_model
