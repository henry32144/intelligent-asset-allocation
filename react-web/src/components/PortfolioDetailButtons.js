import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';
import IconButton from '@material-ui/core/IconButton';
import ShowChartIcon from '@material-ui/icons/ShowChart';
import BarChartIcon from '@material-ui/icons/BarChart';
import SearchIcon from '@material-ui/icons/Search';

const useStyles = makeStyles((theme) => ({
  root: {
    marginLeft: "auto",
    display: 'inline-flex',
  },
}));

export default function PortfolioDetailButtons(props) {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      {props.showSearchButton &&
        <IconButton
          edge="start"
          color="inherit"
          aria-label="expand search stock">
          <SearchIcon />
        </IconButton>
      }
      <IconButton
        edge="start"
        color="inherit"
        aria-label="show performance">
        <ShowChartIcon />
      </IconButton>
      <IconButton
        edge="start"
        color="inherit"
        aria-label="asset weight">
        <BarChartIcon />
      </IconButton>
    </div>
  );
}
