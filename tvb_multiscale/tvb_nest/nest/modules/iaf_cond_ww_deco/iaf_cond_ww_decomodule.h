
/*
 *  iaf_cond_ww_decomodule.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  2019-11-14 14:51:07.526268
 */

#ifndef IAF_COND_WW_DECOMODULE_H
#define IAF_COND_WW_DECOMODULE_H

#include "slimodule.h"
#include "slifunction.h"


/**
* Class defining your model.
* @note For each model, you must define one such class, with a unique name.
*/
class iaf_cond_ww_decomodule : public SLIModule
{
public:
  // Interface functions ------------------------------------------

  /**
   * @note The constructor registers the module with the dynamic loader.
   *       Initialization proper is performed by the init() method.
   */
  iaf_cond_ww_decomodule();

  /**
   * @note The destructor does not do much in modules.
   */
  ~iaf_cond_ww_decomodule();

  /**
   * Initialize module by registering models with the network.
   * @param SLIInterpreter* SLI interpreter
   */
  void init( SLIInterpreter* );

  /**
   * Return the name of your model.
   */
  const std::string name( void ) const;

  /**
   * Return the name of a sli file to execute when iaf_cond_ww_decomodule is loaded.
   * This mechanism can be used to define SLI commands associated with your
   * module, in particular, set up type tries for functions you have defined.
   */
  const std::string commandstring( void ) const;

public:
  // Classes implementing your functions -----------------------------

};

#endif